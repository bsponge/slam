# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/js/cpp/display

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/js/cpp/display/build

# Include any dependencies generated for this target.
include src/CMakeFiles/display.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/display.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/display.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/display.dir/flags.make

src/CMakeFiles/display.dir/main.cpp.o: src/CMakeFiles/display.dir/flags.make
src/CMakeFiles/display.dir/main.cpp.o: ../src/main.cpp
src/CMakeFiles/display.dir/main.cpp.o: src/CMakeFiles/display.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/js/cpp/display/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/display.dir/main.cpp.o"
	cd /home/js/cpp/display/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/display.dir/main.cpp.o -MF CMakeFiles/display.dir/main.cpp.o.d -o CMakeFiles/display.dir/main.cpp.o -c /home/js/cpp/display/src/main.cpp

src/CMakeFiles/display.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/display.dir/main.cpp.i"
	cd /home/js/cpp/display/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/js/cpp/display/src/main.cpp > CMakeFiles/display.dir/main.cpp.i

src/CMakeFiles/display.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/display.dir/main.cpp.s"
	cd /home/js/cpp/display/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/js/cpp/display/src/main.cpp -o CMakeFiles/display.dir/main.cpp.s

# Object files for target display
display_OBJECTS = \
"CMakeFiles/display.dir/main.cpp.o"

# External object files for target display
display_EXTERNAL_OBJECTS =

src/display: src/CMakeFiles/display.dir/main.cpp.o
src/display: src/CMakeFiles/display.dir/build.make
src/display: src/CMakeFiles/display.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/js/cpp/display/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable display"
	cd /home/js/cpp/display/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/display.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/display.dir/build: src/display
.PHONY : src/CMakeFiles/display.dir/build

src/CMakeFiles/display.dir/clean:
	cd /home/js/cpp/display/build/src && $(CMAKE_COMMAND) -P CMakeFiles/display.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/display.dir/clean

src/CMakeFiles/display.dir/depend:
	cd /home/js/cpp/display/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/js/cpp/display /home/js/cpp/display/src /home/js/cpp/display/build /home/js/cpp/display/build/src /home/js/cpp/display/build/src/CMakeFiles/display.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/display.dir/depend
