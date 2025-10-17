cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DiScUtilities)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/DiScUtilities")
set(EXTENSION_CATEGORY "Disc Utilities")
set(EXTENSION_CONTRIBUTORS "Ryan Shores")
set(EXTENSION_DESCRIPTION "This extension is intended to facilitate virtual placement of depth electrode arrays and  better visualize the tissue in contact with the electrodes. Though the features were created with Directional and Scalable (DiSc) electrode arrays in mind, many of the tools can be used on standard probes.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/DiScPlacement.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/DiScUtilities/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(TrajectoryAligner)
add_subdirectory(DiScBuilder)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
