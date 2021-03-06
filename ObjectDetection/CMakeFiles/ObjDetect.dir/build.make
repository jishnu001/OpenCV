# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection

# Include any dependencies generated for this target.
include CMakeFiles/ObjDetect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ObjDetect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ObjDetect.dir/flags.make

CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o: CMakeFiles/ObjDetect.dir/flags.make
CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o: ObjDetect.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o -c /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection/ObjDetect.cpp

CMakeFiles/ObjDetect.dir/ObjDetect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ObjDetect.dir/ObjDetect.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection/ObjDetect.cpp > CMakeFiles/ObjDetect.dir/ObjDetect.cpp.i

CMakeFiles/ObjDetect.dir/ObjDetect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ObjDetect.dir/ObjDetect.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection/ObjDetect.cpp -o CMakeFiles/ObjDetect.dir/ObjDetect.cpp.s

CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o.requires:
.PHONY : CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o.requires

CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o.provides: CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o.requires
	$(MAKE) -f CMakeFiles/ObjDetect.dir/build.make CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o.provides.build
.PHONY : CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o.provides

CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o.provides.build: CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o

# Object files for target ObjDetect
ObjDetect_OBJECTS = \
"CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o"

# External object files for target ObjDetect
ObjDetect_EXTERNAL_OBJECTS =

ObjDetect: CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o
ObjDetect: CMakeFiles/ObjDetect.dir/build.make
ObjDetect: /usr/local/lib/libopencv_videostab.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_ts.a
ObjDetect: /usr/local/lib/libopencv_superres.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_stitching.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_contrib.so.2.4.13
ObjDetect: /usr/lib/x86_64-linux-gnu/libGLU.so
ObjDetect: /usr/lib/x86_64-linux-gnu/libGL.so
ObjDetect: /usr/local/lib/libopencv_nonfree.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_ocl.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_gpu.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_photo.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_objdetect.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_legacy.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_video.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_ml.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_calib3d.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_features2d.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_highgui.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_imgproc.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_flann.so.2.4.13
ObjDetect: /usr/local/lib/libopencv_core.so.2.4.13
ObjDetect: CMakeFiles/ObjDetect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ObjDetect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ObjDetect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ObjDetect.dir/build: ObjDetect
.PHONY : CMakeFiles/ObjDetect.dir/build

CMakeFiles/ObjDetect.dir/requires: CMakeFiles/ObjDetect.dir/ObjDetect.cpp.o.requires
.PHONY : CMakeFiles/ObjDetect.dir/requires

CMakeFiles/ObjDetect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ObjDetect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ObjDetect.dir/clean

CMakeFiles/ObjDetect.dir/depend:
	cd /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection /media/jishnu/Workspace/Linux/Workspace/OpenCV/ObjectDetection/CMakeFiles/ObjDetect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ObjDetect.dir/depend

