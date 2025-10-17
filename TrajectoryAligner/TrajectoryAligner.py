import logging
import os
from typing import Annotated, Optional
import slicer
import vtk
import numpy as np


from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLMarkupsLineNode, vtkMRMLLinearTransformNode, vtkMRMLMarkupsNode


#
# TrajectoryAligner
#

class TrajectoryAligner(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Trajectory Aligner")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "DiSc Utilities")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ryan Shores"]
        self.parent.helpText = _("""
This module aligns virtual devices to trajectory lines by automatically generating and updating transform matrices.
Select or create a line markup to define the trajectory, and optionally link other models or markup objects to follow the alignment.
""")
        self.parent.acknowledgementText = _("""
        Developed for depth probe trajectory alignment workflows.
        """)


#
# TrajectoryAlignerParameterNode
#

@parameterNodeWrapper
class TrajectoryAlignerParameterNode:
    """
    The parameters needed by the module.
    """
    #Create a Line markup node to hold the trajectory
    trajectoryLine: vtkMRMLMarkupsLineNode
    #The Subject hierarchy reference
    trajectoryLineItem: int
    #Create a Linear Transform node to hold the 4x4 matrix transform
    trajectoryTransform: vtkMRMLLinearTransformNode
    #Create an offset to move along the trajectory
    trajectoryOffset: float = 0
    #Create a rotation parameter to hold the device yaw in degrees
    rotationAngle: float = 0
    #Create a Linear Transform node to hold the rotation transform 
    rotationTransform: vtkMRMLLinearTransformNode
    
    #The unit vector of the device prior to import
    deviceUnitVector: str = "[0,0,1]"



#
# TrajectoryAlignerWidget
#

class TrajectoryAlignerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._trajectoryObserverTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/TrajectoryAligner.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class
        self.logic = TrajectoryAlignerLogic()

        # Make sure parameter node is initialized
        self.initializeParameterNode()

        #Get the SubjectHierarchyNode for node manipulation
        self._parameterNode.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Button connections
        self.ui.TrajectoryLineMarkupWidget.markupsSelectorComboBox().connect("currentNodeChanged(vtkMRMLNode*)", self.onTrajectoryLineChanged)

        #Slider connections
        self.ui.rotationSlider.connect("valueChanged(double)",self.onRotationAngleChanged)
        self.ui.trajectoryOffsetSlider.connect("valueChanged(double)",self.onTrajectoryOffsetChanged)

        #Line Unit vector field
        self.ui.deviceUnitLineEdit.connect("textChanged(QString)",self.onDeviceUnitChanged)

        # Select the current line in the selector to whatever is in Trajectory Line
        self.ui.TrajectoryLineMarkupWidget.markupsSelectorComboBox().setCurrentNode(self._parameterNode.trajectoryLine)
        #See if there is a line to add an observer to
        if self._parameterNode.trajectoryLine:
            # Add observer for point modifications
            self._trajectoryObserverTag = self.addObserver(
                    self._parameterNode.trajectoryLine, 
                    slicer.vtkMRMLMarkupsNode.PointModifiedEvent, 
                    self.onTrajectoryPointsModified
                )
        
        #Change the combo box's type to allow visualization of line objects
        self.ui.TrajectoryLineMarkupWidget.markupsSelectorComboBox().nodeTypes = ["vtkMRMLMarkupsLineNode"]
        #Add a filter so only trajectories show up
        #self.ui.TrajectoryLineMarkupWidget.markupsSelectorComboBox().addAttribute("vtkMRMLMarkupsLineNode","Category","Trajectory")
        # Select the current line in the selector
        self.ui.TrajectoryLineMarkupWidget.setCurrentNode(self._parameterNode.trajectoryLine)
        
        #notes
        #Output the current
        #logging.info(self.ui.TrajectoryLineMarkupWidget.currentNode())
        #logging.info(self._parameterNode.trajectoryLine)

        #self.ui.trajectoryControlPointWidget.setMRMLScene(slicer.mrmlScene)
        #self.ui.trajectoryControlPointWidget.setCurrentNode(self._parameterNode.trajectoryLine)


    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()
        if self._trajectoryObserverTag:
            self.removeObserver(self._trajectoryObserverTag)

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.initializeParameterNode()
        # Select the current line in the selector
        self.ui.TrajectoryLineMarkupWidget.setCurrentNode(self._parameterNode.trajectoryLine)

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[TrajectoryAlignerParameterNode]) -> None:
        """Set and observe parameter node."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        
        self._parameterNode = inputParameterNode
        
        if self._parameterNode:
            #self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            #self._parameterNode.connectGui(self.ui) # NOTE: Removed instantiation
            self._updateGUI()

    def _updateGUI(self) -> None:
        """Update GUI elements based on current state."""
        
        #Update the Matrix Viewers' Text
        if self._parameterNode and self._parameterNode.trajectoryLine:
            transformName = f"Transform_to_{self._parameterNode.trajectoryLine.GetName()}"
            rotationName = f"{self._parameterNode.trajectoryLine.GetName()}_PCB_Yaw"
            self.ui.transformNameLabel.text = f"Transform: {transformName}"
            self.ui.transformNameLabel_2.text = f"Rotation: {rotationName}"
        else:
            self.ui.transformNameLabel.text = "Transform: None"
            self.ui.transformNameLabel_2.text = f"Rotation: None"

        #Reconnect the UI Matrix Viewers
        # For Trajectory
        self.ui.trajectoryMatrix.setMRMLTransformNode(self._parameterNode.trajectoryTransform)
        # And Rotation
        #Store reference in widget
        self.ui.rotationMatrix.setMRMLTransformNode(self._parameterNode.rotationTransform)
    
    # #NOTE This is now obsolete, marked for deletion
    # def onCreateLineButton(self) -> None:
    #     """Create a new line markup for trajectory definition."""
    #     self._parameterNode.trajectoryLine = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
    #     #NOTE Commented out to allow procedural naming
    #     #lineNode.SetName("TrajectoryLine")
        
    #     # Add two default points to create a line
    #     self._parameterNode.trajectoryLine.AddControlPoint(vtk.vtkVector3d(0, 0, -50))  # Tip (distal)
    #     self._parameterNode.trajectoryLine.AddControlPoint(vtk.vtkVector3d(0, 0, 50))   # Proximal
        
    #     # Set the line as unlocked for manipulation
    #     self._parameterNode.trajectoryLine.SetLocked(False)
        
    #     # Select this line in the selector
    #     self.ui.trajectoryLineSelector.setCurrentNode(self._parameterNode.trajectoryLine)

    #     '''       
    #     lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
    #     #NOTE Commented out to allow procedural naming
    #     #lineNode.SetName("TrajectoryLine")
        
    #     # Add two default points to create a line
    #     lineNode.AddControlPoint(vtk.vtkVector3d(0, 0, -50))  # Tip (distal)
    #     lineNode.AddControlPoint(vtk.vtkVector3d(0, 0, 50))   # Proximal
        
    #     # Set the line as unlocked for manipulation
    #     lineNode.SetLocked(False)
        
    #     # Select this line in the selector
    #     self.ui.trajectoryLineSelector.setCurrentNode(lineNode)
    #     '''
        
    #     logging.info(f"Created new trajectory line: {self._parameterNode.trajectoryLine.GetName()}")

    def onDeviceUnitChanged(self):
        """Update the device unit parameter"""
        #Set the text in the line edit as the parameter
        self._parameterNode.deviceUnitVector = self.ui.deviceUnitLineEdit.text

        logging.info(f"{self._parameterNode.deviceUnitVector}")
        self.onTrajectoryLineChanged(self._parameterNode.trajectoryLine)

    def onTrajectoryLineChanged(self, node) -> None:
        """Handle trajectory line selection change."""
        #Handle the previously active line before switching
        if self._parameterNode and self._parameterNode.trajectoryLine:
            #Remove the previous observer
            self.removeObserver(self._parameterNode.trajectoryLine, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onTrajectoryPointsModified)
            #Lock the previously active line
            self._parameterNode.trajectoryLine.SetLocked(True)

        #Handle the new line
        if node: 
            #If the module has a parameter node with a set trajectory line
            if self._parameterNode:
                #NOTE: Make the current node the trajectory line (assuming it is a line)
                self._parameterNode.trajectoryLine = node
                #If creating a new line (which has no control points)
                if self._parameterNode.trajectoryLine.GetNumberOfControlPoints() < 2:
                    #Name it
                    self._parameterNode.trajectoryLine.SetName("Device_Trajectory")
                    # Add two default points to create a line
                    self._parameterNode.trajectoryLine.AddControlPoint(vtk.vtkVector3d(0, 0, 0), "Tip")  # Tip (distal)
                    self._parameterNode.trajectoryLine.AddControlPoint(vtk.vtkVector3d(0, 0, 10), "Shaft")   # Proximal
                    logging.info(f"Added 2 control points to {self._parameterNode.trajectoryLine.GetName()}")
                # Set the line as unlocked for manipulation
                self._parameterNode.trajectoryLine.SetLocked(False)
            
            # Add observer for point modifications
            self._trajectoryObserverTag = self.addObserver(
                self._parameterNode.trajectoryLine, 
                slicer.vtkMRMLMarkupsNode.PointModifiedEvent, 
                self.onTrajectoryPointsModified
            )


            '''
            logging.info(f"Trajectory Observer")      
            # Also observe when points are added/removed
            self._trajectoryPointAddedObserverTag = self.addObserver(
                self._parameterNode.trajectoryLine, 
                slicer.vtkMRMLMarkupsNode.PointAddedEvent, 
                self.onTrajectoryPointsModified)
            
            self._trajectoryPointRemovedObserverTag = self.addObserver(
                self._parameterNode.trajectoryLine, 
                slicer.vtkMRMLMarkupsNode.PointRemovedEvent, 
                self.onTrajectoryPointsModified)
            '''
            logging.info(f"Selected trajectory line: {node.GetName()}")
            
            # Update transform immediately
            if self._parameterNode:
                self.logic.updateTransformFromTrajectory(node, self._parameterNode)
                #And update rotation
                self.logic.updateRotationFromTrajectory(self._parameterNode)
                  
        self._updateGUI()

    def onTrajectoryPointsModified(self, caller, event) -> None:
        """Handle trajectory line point modifications."""
        if self._parameterNode and self._parameterNode.trajectoryLine:
            self.logic.updateTransformFromTrajectory(self._parameterNode.trajectoryLine, self._parameterNode)

    def onRotationAngleChanged(self,value):
        """Handle the implant rotation parameter (yaw)"""
        if self._parameterNode:
            #Store the value in the parameter node
            self._parameterNode.rotationAngle = value
            #Update the matrix
            #self.logic.updateRotationFromTrajectory(parameterNode=self._parameterNode)
            #Use the master transform computation now instead of the separate rotation one
            self.logic.updateTransformFromTrajectory(self._parameterNode.trajectoryLine, self._parameterNode)

    def onTrajectoryOffsetChanged(self, value):
        """Handle the implant rotation parameter (yaw)"""
        if self._parameterNode:
            #Store the value in the parameter node
            self._parameterNode.trajectoryOffset = value
            #Update the matrix
            #self.logic.updateRotationFromTrajectory(parameterNode=self._parameterNode)
            #Use the master transform computation now instead of the separate rotation one
            self.logic.updateTransformFromTrajectory(self._parameterNode.trajectoryLine, self._parameterNode)



#
# TrajectoryAlignerLogic
#

class TrajectoryAlignerLogic(ScriptedLoadableModuleLogic):
    """This class implements the actual computation for trajectory alignment."""

    def __init__(self) -> None:
        """Called when the logic class is instantiated."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return TrajectoryAlignerParameterNode(super().getParameterNode())

    def find_transform_matrix(self, trajectory_start: np.ndarray, trajectory_end: np.ndarray, 
                            device_unit: np.ndarray = np.array([0, 0, 1])) -> tuple:
        """Generate the transform matrix to align device_unit vector to trajectory.
        
        Args:
            trajectory_start: The distal coordinate of the probe (deepest in tissue)
            trajectory_end: The proximal coordinate of the probe (closest to surface)  
            device_unit: The unit vector that the virtual device coordinates are mapped to
            
        Returns:
            tuple: (R, trajectory_unit) - rotation matrix and trajectory unit vector
        """
        
        # Find the vector between trajectory points
        trajectory = trajectory_end - trajectory_start
        
        # Find the unit vector of the trajectory
        trajectory_unit = trajectory / np.linalg.norm(trajectory)
        
        # Specify the input vector and target vector
        a = device_unit
        b = trajectory_unit
        
        # Calculate the cross product
        v = np.cross(a, b)
        # The norm of the cross product
        s = np.linalg.norm(v)
        # And the dot product
        c = np.dot(a, b)
        
        # Handle the case where vectors are already aligned
        if s < 1e-10:  # vectors are parallel
            if c > 0:  # same direction
                R = np.eye(3)
            else:  # opposite direction
                R = -np.eye(3)
        else:
            # Generate a skew-symmetric matrix of the cross product
            kmat = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            
            # Define the rotation matrix R using Rodrigues' rotation formula
            R = np.eye(3) + kmat + (kmat.dot(kmat) * ((1 - c) / (s**2)))

        return R, trajectory_unit

    def updateTransformFromTrajectory(self, trajectoryLine: vtkMRMLMarkupsLineNode, 
                                    parameterNode: TrajectoryAlignerParameterNode) -> None:
        """Update or create transform matrix based on trajectory line.
        
        Args:
            trajectoryLine: The line markup defining the trajectory
            parameterNode: Parameter node containing settings
        """
        
        if not trajectoryLine or trajectoryLine.GetNumberOfControlPoints() < 2:
            logging.warning("Trajectory line must have at least 2 control points")
            return
        
        # Get trajectory points
        traj_points = slicer.util.arrayFromMarkupsControlPoints(trajectoryLine)
        
        if traj_points.shape[0] < 2:
            logging.warning("Could not get trajectory points")
            return
        
        # Parse device unit vector from string
        try:
            device_unit_str = parameterNode.deviceUnitVector.strip('[]')
            device_unit = np.array([float(x.strip()) for x in device_unit_str.split(',')])
        except:
            device_unit = np.array([0, 0, 1])  # Default to z-axis
            logging.warning("Could not parse device unit vector, using default [0,0,1]")
        
        # Generate transform matrix
        R, trajectory_unit = self.find_transform_matrix(
            traj_points[0, :],  # First point (distal/tip)
            traj_points[1, :],  # Second point (proximal)
            device_unit
        )
        
        # Create 4x4 transform matrix r hat which 
        Rhat = np.eye(4)
        Rhat[:3, :3] = R
        Rhat[:3, 3] = traj_points[0, :]  # Translation to first point

        #Define a second rotation matrix based on the yaw angle
        #Get the new rotation angle in radians
        #theta = (self.getParameterNode.rotationAngle/180) * (np.pi)
        parameterNode = self.getParameterNode()
        print(parameterNode)
        theta = (parameterNode.rotationAngle/180) * (np.pi)
        #Use theta to calculate the rotation matrix
        Y = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, parameterNode.trajectoryOffset],
            [0, 0, 0, 1]        
            ])

        #Combine the two transforms by multiplying in reverse order (Yaw then trajectory so trajectory @ yaw)
        transform_matrix = Rhat @ Y 
        
        # Get or create transform node
        transform_name = f"Transform_to_{trajectoryLine.GetName()}"
        parameterNode.trajectoryTransform = slicer.mrmlScene.GetFirstNodeByName(transform_name)
        
        if parameterNode.trajectoryTransform is None:
            parameterNode.trajectoryTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
            parameterNode.trajectoryTransform.SetName(transform_name)
            #Set the trajectory line as the parent
            #parameterNode.shNode.SetItemParent(parameterNode.trajectoryTransform, parameterNode.shNode.GetItemByDataNode(trajectoryLine))
            logging.info(f"Created new transform node: {transform_name}")
        
        # Update the transform matrix
        parameterNode.trajectoryTransform.SetMatrixTransformToParent( #NOTE Changed rom SetAndObserveMatrixTransformToParent since deprecated
            slicer.util.vtkMatrixFromArray(transform_matrix)
        )
        
        #logging.info(f"Updated transform matrix for trajectory: {trajectoryLine.GetName()}")
    
    def updateRotationFromTrajectory(self, parameterNode: TrajectoryAlignerParameterNode) -> None:
        #See if there is an active trajectory
        if parameterNode.trajectoryLine:
            #Get the new rotation angle in radians
            theta = (parameterNode.rotationAngle/180) * (np.pi)
            #Use theta to calculate the rotation matrix
            transform_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta),0, 0],
                [0, 0, 1,parameterNode.trajectoryOffset],
                [0,0,0,1]        
                ])
            
            # # Create 4x4 transform matrix by adding an identity row and column
            # transform_matrix = np.eye(4)
            # transform_matrix[:3, :3] = R

            #Link to node
            transform_name = parameterNode.trajectoryLine.GetName().replace("_Trajectory","_PCB_Yaw")
            parameterNode.rotationTransform = slicer.mrmlScene.GetFirstNodeByName(transform_name)
            if parameterNode.rotationTransform is None:
                parameterNode.rotationTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
                parameterNode.rotationTransform.SetName(transform_name)
                logging.info(f"Created new transform node: {transform_name}")
            
            # Update the transform matrix
            parameterNode.rotationTransform.SetMatrixTransformToParent( #NOTE Changed rom SetAndObserveMatrixTransformToParent since deprecated
                slicer.util.vtkMatrixFromArray(transform_matrix)
            )




#
# TrajectoryAlignerTest
#

class TrajectoryAlignerTest(ScriptedLoadableModuleTest):
    """Test case for the scripted module."""

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_TrajectoryAligner1()

    def test_TrajectoryAligner1(self):
        """Test basic functionality."""
        
        self.delayDisplay("Starting the test")
        
        # Create test trajectory line
        lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        lineNode.SetName("TestTrajectory")
        lineNode.AddControlPoint(vtk.vtkVector3d(0, 0, -50))
        lineNode.AddControlPoint(vtk.vtkVector3d(0, 0, 50))
        
        # Test the module logic
        logic = TrajectoryAlignerLogic()
        parameterNode = logic.getParameterNode()
        parameterNode.trajectoryLine = lineNode
        
        # Update transform
        logic.updateTransformFromTrajectory(lineNode, parameterNode)
        
        # Check that transform was created
        transformName = f"Transform_to_{lineNode.GetName()}"
        trajectoryTransform = slicer.mrmlScene.GetFirstNodeByName(transformName)
        self.assertIsNotNone(trajectoryTransform)
        
        self.delayDisplay("Test passed")