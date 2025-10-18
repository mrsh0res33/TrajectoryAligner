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

from slicer import vtkMRMLMarkupsLineNode, vtkMRMLLinearTransformNode, vtkMRMLMarkupsNode, vtkMRMLSubjectHierarchyNode


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
    #The subject hierarchy node for the scene
    shNode: vtkMRMLSubjectHierarchyNode
    #Create a Line markup node to hold the trajectory
    trajectoryLine: vtkMRMLMarkupsLineNode
    #Create a Linear Transform node to hold the 4x4 matrix transform
    trajectoryTransform: vtkMRMLLinearTransformNode
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

        # --Connections--
        
        # Scene Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Button connections
        self.ui.TrajectoryLineMarkupWidget.markupsSelectorComboBox().connect("currentNodeChanged(vtkMRMLNode*)", self.onTrajectoryLineChanged)

        #Slider connections
        self.ui.rotationSlider.connect("valueChanged(double)",self.onRotationAngleChanged) #Twist rotation
        self.ui.trajectoryOffsetSlider.connect("valueChanged(double)",self.onTrajectoryOffsetChanged) #Move along trajectory

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
        #Update the transform matrix accordingly
        self.updateLinkedMatrix()


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
        if self._parameterNode and self._parameterNode.trajectoryTransform:
            self.ui.transformNameLabel.text = f"{self._parameterNode.trajectoryTransform.GetName()}"
        else:
            self.ui.transformNameLabel.text = "No Line Selected"

        #Connect the UI Matrix Viewer for Transform matrix
        self.ui.trajectoryMatrix.setMRMLTransformNode(self._parameterNode.trajectoryTransform)


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
                    self._parameterNode.shNode.SetItemName(self._parameterNode.shNode.GetItemByDataNode(self._parameterNode.trajectoryLine),"New Trajectory")
                    #Get the display node
                    displayNode = self._parameterNode.trajectoryLine.GetDisplayNode()
                    #Set it to unconstrained snap (index 0)
                    displayNode.SetSnapMode(0)
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
            
            #Log that a new line has been selected
            logging.info(f"Selected trajectory line: {node.GetName()}")
            
            #Update the linked Matrix
            self.updateLinkedMatrix()
            
            # Update transform immediately
            if self._parameterNode:
                self.logic.updateTransformFromTrajectory()
        #If selection set to none
        else:
            #Clear the parameter
            self._parameterNode.trajectoryLine = None
            #Update the linked Matrix
            self.updateLinkedMatrix()

        #Update the GUI either way  
        self._updateGUI()

    def onTrajectoryPointsModified(self, caller, event) -> None:
        """Handle trajectory line point modifications."""
        if self._parameterNode and self._parameterNode.trajectoryLine:
            self.logic.updateTransformFromTrajectory()

    def onRotationAngleChanged(self,value):
        """Handle the implant rotation parameter (yaw/twist)"""
        if self._parameterNode and self._parameterNode.trajectoryTransform:
            #Instead of storing the value in the parameter node, store it as an attribute in the transform matrix
            self._parameterNode.trajectoryTransform.SetAttribute("rotationAngle", str(value))
            #Use the master transform computation now instead of the separate rotation one
            self.logic.updateTransformFromTrajectory()

    def onTrajectoryOffsetChanged(self, value):
        """Handle the implant trajectory offset (displacement along trajectory)"""
        if self._parameterNode and self._parameterNode.trajectoryTransform:
            #Instead of storing the value in the parameter node, store it as an attribute in the transform matrix
            self._parameterNode.trajectoryTransform.SetAttribute("trajectoryOffset", str(value))
            #Use the master transform computation now instead of the separate rotation one
            self.logic.updateTransformFromTrajectory()

    def updateLinkedMatrix(self):
        """Update the linked matrix (trajectory transform parameter) based on the current trajectory line"""
        #If there is a trajectoryLine selected 
        if self._parameterNode.trajectoryLine:
            # -- Look for linked matrix --
            #Get the item ID of the current line
            trajectoryLineItem = self._parameterNode.shNode.GetItemByDataNode(self._parameterNode.trajectoryLine)
            #Make a list to hold the output
            childIDs = vtk.vtkIdList()
            #Attempt to get the item's linked matrix (should be a child)
            self._parameterNode.shNode.GetItemChildren(trajectoryLineItem,childIDs)
            #Output variable for search
            foundLinkedArray = False
            #Check all children
            for i in range(childIDs.GetNumberOfIds()):
                #Get the node ID
                childItemID = childIDs.GetId(i)
                #Get the node's category and look for a linked matrix
                childItemCategory = self._parameterNode.shNode.GetItemAttribute(childItemID, "Category")
                #If It's a linked matrix
                if (childItemCategory == "LinkedMatrix"):
                    #Get the child node from the ID and store it in parameter node
                    self._parameterNode.trajectoryTransform = self._parameterNode.shNode.GetItemDataNode(childItemID)
                    #Denote that we found something
                    foundLinkedArray = True
                    #Update the name if necessary
                    #If the name doesn't match its parent
                    if self._parameterNode.shNode.GetItemName(childItemID) != f"Linked Matrix ({self._parameterNode.trajectoryLine.GetName()})":
                        #Rename the linked matrix
                        self._parameterNode.shNode.SetItemName(childItemID, f"Linked Matrix ({self._parameterNode.trajectoryLine.GetName()})")
                        #Log the change
                        logging.info(f"Renamed Linked Transform to match parent [{self._parameterNode.shNode.GetItemName(childItemID)}]")    
                    #Print success to log
                    logging.info(f"Found Linked Matrix node: {self._parameterNode.trajectoryTransform.GetName()}")
                    break
            #--If it doesn't exist,--
            if foundLinkedArray == False:
                #Make it
                self._parameterNode.trajectoryTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
                #Add the fine tune attributes
                self._parameterNode.trajectoryTransform.SetAttribute("rotationAngle", "0") #Rotation
                self._parameterNode.trajectoryTransform.SetAttribute("trajectoryOffset", "0") #Offset
                #Define Transform name
                transform_name = f"Linked Matrix ({self._parameterNode.trajectoryLine.GetName()})"
                #Get a reference to the item
                trajectoryTransformItem = self._parameterNode.shNode.GetItemByDataNode(self._parameterNode.trajectoryTransform)
                #Set its name
                #self._parameterNode.trajectoryTransform.SetName(transform_name)
                self._parameterNode.shNode.SetItemName(trajectoryTransformItem, transform_name)
                #Also set an attribute for the item in the hierarchy
                self._parameterNode.shNode.SetItemAttribute(trajectoryTransformItem, "Category", "LinkedMatrix")
                #Set the trajectory line as the parent
                self._parameterNode.shNode.SetItemParent(trajectoryTransformItem, self._parameterNode.shNode.GetItemByDataNode(self._parameterNode.trajectoryLine))
                logging.info(f"Created new Linked Matrix: {transform_name}")

            #Load the fine tune attributes
            #If there is a stored rotationAngle parameter
            if self._parameterNode.trajectoryTransform and self._parameterNode.trajectoryTransform.GetAttribute("rotationAngle"):
                #Load it to the UI
                self.ui.rotationSlider.setValue(float(self._parameterNode.trajectoryTransform.GetAttribute("rotationAngle"))) #Twist rotation
            #If not
            else:
                #Initialize the parameter
                self._parameterNode.trajectoryTransform.SetAttribute("rotationAngle", "0")
                #Reset the slider to 0
                self.ui.rotationSlider.setValue(0) #Twist rotation
            #If there is a stored trajectoryOffset parameter    
            if self._parameterNode.trajectoryTransform and self._parameterNode.trajectoryTransform.GetAttribute("trajectoryOffset"):
                #Load it to the UI
                self.ui.trajectoryOffsetSlider.setValue(float(self._parameterNode.trajectoryTransform.GetAttribute("trajectoryOffset")))#Move along trajectory
            #If not,
            else:
                #Initialize the parameter
                self._parameterNode.trajectoryTransform.SetAttribute("trajectoryOffset", "0")
                #Reset the slider to 0
                self.ui.trajectoryOffsetSlider.setValue(0)#Move along trajectory

        #If there isn't a trajectory line
        else:
            #Clear the trajectory transform as well
            self._parameterNode.trajectoryTransform = None
            #Clear the fine tune attributes
            self.ui.rotationSlider.setValue(0) #Twist rotation
            self.ui.trajectoryOffsetSlider.setValue(0)#Move along trajectory

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

    def updateTransformFromTrajectory(self) -> None:
        """Update or create transform matrix based on trajectory line.
        
        Args:
            trajectoryLine: The line markup defining the trajectory
            parameterNode: Parameter node containing settings
        """

        #Get the extension's parameter node
        parameterNode = self.getParameterNode()
        #Get the trajectory line from the parameter node
        trajectoryLine = parameterNode.trajectoryLine
        
        if not trajectoryLine or trajectoryLine.GetNumberOfControlPoints() < 2:
            logging.warning("Trajectory line must have at least 2 control points")
            return
        
        # Get trajectory points
        traj_points = slicer.util.arrayFromMarkupsControlPoints(trajectoryLine)
        
        if traj_points.shape[0] < 2:
            logging.warning("Could not get trajectory points")
            return
        
        # Parse device unit vector from string in parameter node
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
        
        #Create 4x4 transform matrix Rhat which can also process translation 
        Rhat = np.eye(4)
        Rhat[:3, :3] = R
        Rhat[:3, 3] = traj_points[0, :]  # Translation to first point

        #Define a second rotation matrix based on the yaw angle

        #Check the matrix's attributes for an offset parameter
        if parameterNode.trajectoryTransform and parameterNode.trajectoryTransform.GetAttribute("trajectoryOffset"):
            #If found, use it
            offset =  float(parameterNode.trajectoryTransform.GetAttribute("trajectoryOffset"))
        #If none
        else:
            #Default to zero so the calculation works
            offset = 0

        #Check the matrix's attributes for a rotation parameter
        if parameterNode.trajectoryTransform and parameterNode.trajectoryTransform.GetAttribute("rotationAngle"):
            rotationAngle = float(parameterNode.trajectoryTransform.GetAttribute("rotationAngle"))
            #If found, use it to get the new rotation angle in radians
            theta = (rotationAngle/180) * (np.pi)     
        #If none
        else:
            #Default to zero so the calculation works
            theta = 0

        #Use theta and offset to calculate the rotation matrix
        Y = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, offset],
            [0, 0, 0, 1]        
            ])

        #Combine the two transforms by multiplying in reverse order (Yaw then trajectory so trajectory @ yaw)
        transform_matrix = Rhat @ Y 
        
        # Update the transform node with new transform matrix
        parameterNode.trajectoryTransform.SetMatrixTransformToParent( #NOTE Changed rom SetAndObserveMatrixTransformToParent since deprecated
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
        logic.updateTransformFromTrajectory()
        
        # Check that transform was created
        transformName = f"Transform_to_{lineNode.GetName()}"
        trajectoryTransform = slicer.mrmlScene.GetFirstNodeByName(transformName)
        self.assertIsNotNone(trajectoryTransform)
        
        self.delayDisplay("Test passed")