# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mjl152/Documents/stuff/greplace

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mjl152/Documents/stuff/greplace

# Include any dependencies generated for this target.
include CMakeFiles/greplace.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/greplace.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/greplace.dir/flags.make

CMakeFiles/greplace.dir/main.cpp.o: CMakeFiles/greplace.dir/flags.make
CMakeFiles/greplace.dir/main.cpp.o: main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/mjl152/Documents/stuff/greplace/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/greplace.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/greplace.dir/main.cpp.o -c /home/mjl152/Documents/stuff/greplace/main.cpp

CMakeFiles/greplace.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/greplace.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/mjl152/Documents/stuff/greplace/main.cpp > CMakeFiles/greplace.dir/main.cpp.i

CMakeFiles/greplace.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/greplace.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/mjl152/Documents/stuff/greplace/main.cpp -o CMakeFiles/greplace.dir/main.cpp.s

CMakeFiles/greplace.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/greplace.dir/main.cpp.o.requires

CMakeFiles/greplace.dir/main.cpp.o.provides: CMakeFiles/greplace.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/greplace.dir/build.make CMakeFiles/greplace.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/greplace.dir/main.cpp.o.provides

CMakeFiles/greplace.dir/main.cpp.o.provides.build: CMakeFiles/greplace.dir/main.cpp.o

CMakeFiles/greplace.dir/cpu.cpp.o: CMakeFiles/greplace.dir/flags.make
CMakeFiles/greplace.dir/cpu.cpp.o: cpu.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/mjl152/Documents/stuff/greplace/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/greplace.dir/cpu.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/greplace.dir/cpu.cpp.o -c /home/mjl152/Documents/stuff/greplace/cpu.cpp

CMakeFiles/greplace.dir/cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/greplace.dir/cpu.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/mjl152/Documents/stuff/greplace/cpu.cpp > CMakeFiles/greplace.dir/cpu.cpp.i

CMakeFiles/greplace.dir/cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/greplace.dir/cpu.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/mjl152/Documents/stuff/greplace/cpu.cpp -o CMakeFiles/greplace.dir/cpu.cpp.s

CMakeFiles/greplace.dir/cpu.cpp.o.requires:
.PHONY : CMakeFiles/greplace.dir/cpu.cpp.o.requires

CMakeFiles/greplace.dir/cpu.cpp.o.provides: CMakeFiles/greplace.dir/cpu.cpp.o.requires
	$(MAKE) -f CMakeFiles/greplace.dir/build.make CMakeFiles/greplace.dir/cpu.cpp.o.provides.build
.PHONY : CMakeFiles/greplace.dir/cpu.cpp.o.provides

CMakeFiles/greplace.dir/cpu.cpp.o.provides.build: CMakeFiles/greplace.dir/cpu.cpp.o

CMakeFiles/greplace.dir/person.cpp.o: CMakeFiles/greplace.dir/flags.make
CMakeFiles/greplace.dir/person.cpp.o: person.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/mjl152/Documents/stuff/greplace/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/greplace.dir/person.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/greplace.dir/person.cpp.o -c /home/mjl152/Documents/stuff/greplace/person.cpp

CMakeFiles/greplace.dir/person.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/greplace.dir/person.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/mjl152/Documents/stuff/greplace/person.cpp > CMakeFiles/greplace.dir/person.cpp.i

CMakeFiles/greplace.dir/person.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/greplace.dir/person.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/mjl152/Documents/stuff/greplace/person.cpp -o CMakeFiles/greplace.dir/person.cpp.s

CMakeFiles/greplace.dir/person.cpp.o.requires:
.PHONY : CMakeFiles/greplace.dir/person.cpp.o.requires

CMakeFiles/greplace.dir/person.cpp.o.provides: CMakeFiles/greplace.dir/person.cpp.o.requires
	$(MAKE) -f CMakeFiles/greplace.dir/build.make CMakeFiles/greplace.dir/person.cpp.o.provides.build
.PHONY : CMakeFiles/greplace.dir/person.cpp.o.provides

CMakeFiles/greplace.dir/person.cpp.o.provides.build: CMakeFiles/greplace.dir/person.cpp.o

# Object files for target greplace
greplace_OBJECTS = \
"CMakeFiles/greplace.dir/main.cpp.o" \
"CMakeFiles/greplace.dir/cpu.cpp.o" \
"CMakeFiles/greplace.dir/person.cpp.o"

# External object files for target greplace
greplace_EXTERNAL_OBJECTS =

greplace: CMakeFiles/greplace.dir/main.cpp.o
greplace: CMakeFiles/greplace.dir/cpu.cpp.o
greplace: CMakeFiles/greplace.dir/person.cpp.o
greplace: CMakeFiles/greplace.dir/build.make
greplace: /usr/lib64/libopencv_calib3d.so
greplace: /usr/lib64/libopencv_contrib.so
greplace: /usr/lib64/libopencv_core.so
greplace: /usr/lib64/libopencv_features2d.so
greplace: /usr/lib64/libopencv_flann.so
greplace: /usr/lib64/libopencv_highgui.so
greplace: /usr/lib64/libopencv_imgproc.so
greplace: /usr/lib64/libopencv_legacy.so
greplace: /usr/lib64/libopencv_ml.so
greplace: /usr/lib64/libopencv_objdetect.so
greplace: /usr/lib64/libopencv_photo.so
greplace: /usr/lib64/libopencv_stitching.so
greplace: /usr/lib64/libopencv_superres.so
greplace: /usr/lib64/libopencv_ts.so
greplace: /usr/lib64/libopencv_video.so
greplace: /usr/lib64/libopencv_videostab.so
greplace: CMakeFiles/greplace.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable greplace"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/greplace.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/greplace.dir/build: greplace
.PHONY : CMakeFiles/greplace.dir/build

CMakeFiles/greplace.dir/requires: CMakeFiles/greplace.dir/main.cpp.o.requires
CMakeFiles/greplace.dir/requires: CMakeFiles/greplace.dir/cpu.cpp.o.requires
CMakeFiles/greplace.dir/requires: CMakeFiles/greplace.dir/person.cpp.o.requires
.PHONY : CMakeFiles/greplace.dir/requires

CMakeFiles/greplace.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/greplace.dir/cmake_clean.cmake
.PHONY : CMakeFiles/greplace.dir/clean

CMakeFiles/greplace.dir/depend:
	cd /home/mjl152/Documents/stuff/greplace && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mjl152/Documents/stuff/greplace /home/mjl152/Documents/stuff/greplace /home/mjl152/Documents/stuff/greplace /home/mjl152/Documents/stuff/greplace /home/mjl152/Documents/stuff/greplace/CMakeFiles/greplace.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/greplace.dir/depend

