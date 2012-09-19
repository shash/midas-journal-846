# This is the root ITK CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)


# This project is designed to be built outside the Insight source tree.
PROJECT( PhaseSymmetryImageFilter)

# Find ITK.
FIND_PACKAGE(ITK REQUIRED)
FIND_PACKAGE(WrapITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

set(PHASE_SYMMETRY_HEADERS itkButterworthFilterFreqImageSource.h itkLogGaborFreqImageSource.h itkPhaseSymmetryImageFilter.h itkRealAndImaginaryToComplexImageFilter.h itkSteerableFilterFreqImageSource.h)
set(PHASE_SYMMETRY_TEMPLATES itkButterworthFilterFreqImageSource.txx itkLogGaborFreqImageSource.txx itkPhaseSymmetryImageFilter.txx itkSteerableFilterFreqImageSource.txx)


ADD_EXECUTABLE( PhaseSymmetryImageFilter2D main2.cxx ${PHASE_SYMMETRY_HEADERS} ${PHASE_SYMMETRY_TEMPLATES})
ADD_EXECUTABLE( PhaseSymmetryImageFilter3D main3.cxx ${PHASE_SYMMETRY_HEADERS} ${PHASE_SYMMETRY_TEMPLATES})

TARGET_LINK_LIBRARIES( PhaseSymmetryImageFilter3D ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES( PhaseSymmetryImageFilter2D ${ITK_LIBRARIES})

itk_wrap_module("${PROJECT_NAME}")
	set(WRAPPER_LIBRARY_GROUPS
		itkPhaseSymmetryImageFilter
		)
  set(WRAPPER_LIBRARY_DEPENDS ITKImageFilterBase)
  itk_auto_load_submodules()

itk_end_wrap_module()