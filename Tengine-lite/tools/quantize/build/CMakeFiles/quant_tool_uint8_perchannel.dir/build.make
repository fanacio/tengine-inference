# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/tools/Tengine-tengine-lite/tools/quantize

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tools/Tengine-tengine-lite/tools/quantize/build

# Include any dependencies generated for this target.
include CMakeFiles/quant_tool_uint8_perchannel.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/quant_tool_uint8_perchannel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/quant_tool_uint8_perchannel.dir/flags.make

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o: CMakeFiles/quant_tool_uint8_perchannel.dir/flags.make
CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o: ../quant_save_graph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o -c /home/tools/Tengine-tengine-lite/tools/quantize/quant_save_graph.cpp

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tools/Tengine-tengine-lite/tools/quantize/quant_save_graph.cpp > CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.i

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tools/Tengine-tengine-lite/tools/quantize/quant_save_graph.cpp -o CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.s

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o.requires:

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o.requires

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o.provides: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o.requires
	$(MAKE) -f CMakeFiles/quant_tool_uint8_perchannel.dir/build.make CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o.provides.build
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o.provides

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o.provides.build: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o


CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o: CMakeFiles/quant_tool_uint8_perchannel.dir/flags.make
CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o: ../algorithm/quant_dfq.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o -c /home/tools/Tengine-tengine-lite/tools/quantize/algorithm/quant_dfq.cpp

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tools/Tengine-tengine-lite/tools/quantize/algorithm/quant_dfq.cpp > CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.i

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tools/Tengine-tengine-lite/tools/quantize/algorithm/quant_dfq.cpp -o CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.s

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o.requires:

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o.requires

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o.provides: CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o.requires
	$(MAKE) -f CMakeFiles/quant_tool_uint8_perchannel.dir/build.make CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o.provides.build
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o.provides

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o.provides.build: CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o


CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o: CMakeFiles/quant_tool_uint8_perchannel.dir/flags.make
CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o: ../algorithm/quant_eq.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o -c /home/tools/Tengine-tengine-lite/tools/quantize/algorithm/quant_eq.cpp

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tools/Tengine-tengine-lite/tools/quantize/algorithm/quant_eq.cpp > CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.i

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tools/Tengine-tengine-lite/tools/quantize/algorithm/quant_eq.cpp -o CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.s

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o.requires:

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o.requires

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o.provides: CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o.requires
	$(MAKE) -f CMakeFiles/quant_tool_uint8_perchannel.dir/build.make CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o.provides.build
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o.provides

CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o.provides.build: CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o


CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o: CMakeFiles/quant_tool_uint8_perchannel.dir/flags.make
CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o: ../quant_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o -c /home/tools/Tengine-tengine-lite/tools/quantize/quant_utils.cpp

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tools/Tengine-tengine-lite/tools/quantize/quant_utils.cpp > CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.i

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tools/Tengine-tengine-lite/tools/quantize/quant_utils.cpp -o CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.s

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o.requires:

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o.requires

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o.provides: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/quant_tool_uint8_perchannel.dir/build.make CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o.provides.build
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o.provides

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o.provides.build: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o


CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o: CMakeFiles/quant_tool_uint8_perchannel.dir/flags.make
CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o: /home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o -c /home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp > CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.i

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp -o CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.s

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o.requires:

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o.requires

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o.provides: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o.requires
	$(MAKE) -f CMakeFiles/quant_tool_uint8_perchannel.dir/build.make CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o.provides.build
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o.provides

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o.provides.build: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o


CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o: CMakeFiles/quant_tool_uint8_perchannel.dir/flags.make
CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o: /home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o -c /home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp > CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.i

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp -o CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.s

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o.requires:

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o.requires

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o.provides: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o.requires
	$(MAKE) -f CMakeFiles/quant_tool_uint8_perchannel.dir/build.make CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o.provides.build
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o.provides

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o.provides.build: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o


CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o: CMakeFiles/quant_tool_uint8_perchannel.dir/flags.make
CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o: /home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o   -c /home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c > CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.i

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c -o CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.s

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o.requires:

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o.requires

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o.provides: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o.requires
	$(MAKE) -f CMakeFiles/quant_tool_uint8_perchannel.dir/build.make CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o.provides.build
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o.provides

CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o.provides.build: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o


CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o: CMakeFiles/quant_tool_uint8_perchannel.dir/flags.make
CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o: ../quant_tool_uint8_perchannel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o -c /home/tools/Tengine-tengine-lite/tools/quantize/quant_tool_uint8_perchannel.cpp

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tools/Tengine-tengine-lite/tools/quantize/quant_tool_uint8_perchannel.cpp > CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.i

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tools/Tengine-tengine-lite/tools/quantize/quant_tool_uint8_perchannel.cpp -o CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.s

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o.requires:

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o.requires

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o.provides: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o.requires
	$(MAKE) -f CMakeFiles/quant_tool_uint8_perchannel.dir/build.make CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o.provides.build
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o.provides

CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o.provides.build: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o


# Object files for target quant_tool_uint8_perchannel
quant_tool_uint8_perchannel_OBJECTS = \
"CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o" \
"CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o" \
"CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o" \
"CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o" \
"CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o" \
"CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o" \
"CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o" \
"CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o"

# External object files for target quant_tool_uint8_perchannel
quant_tool_uint8_perchannel_EXTERNAL_OBJECTS =

quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o
quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o
quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o
quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o
quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o
quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o
quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o
quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o
quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/build.make
quant_tool_uint8_perchannel: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
quant_tool_uint8_perchannel: /usr/local/lib/libopencv_imgproc.so.3.2.0
quant_tool_uint8_perchannel: /usr/local/lib/libopencv_core.so.3.2.0
quant_tool_uint8_perchannel: CMakeFiles/quant_tool_uint8_perchannel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable quant_tool_uint8_perchannel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quant_tool_uint8_perchannel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/quant_tool_uint8_perchannel.dir/build: quant_tool_uint8_perchannel

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/build

CMakeFiles/quant_tool_uint8_perchannel.dir/requires: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_save_graph.cpp.o.requires
CMakeFiles/quant_tool_uint8_perchannel.dir/requires: CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_dfq.cpp.o.requires
CMakeFiles/quant_tool_uint8_perchannel.dir/requires: CMakeFiles/quant_tool_uint8_perchannel.dir/algorithm/quant_eq.cpp.o.requires
CMakeFiles/quant_tool_uint8_perchannel.dir/requires: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_utils.cpp.o.requires
CMakeFiles/quant_tool_uint8_perchannel.dir/requires: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/save_graph.cpp.o.requires
CMakeFiles/quant_tool_uint8_perchannel.dir/requires: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_op_save.cpp.o.requires
CMakeFiles/quant_tool_uint8_perchannel.dir/requires: CMakeFiles/quant_tool_uint8_perchannel.dir/home/tools/Tengine-tengine-lite/tools/save_graph/tm2_generate.c.o.requires
CMakeFiles/quant_tool_uint8_perchannel.dir/requires: CMakeFiles/quant_tool_uint8_perchannel.dir/quant_tool_uint8_perchannel.cpp.o.requires

.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/requires

CMakeFiles/quant_tool_uint8_perchannel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/quant_tool_uint8_perchannel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/clean

CMakeFiles/quant_tool_uint8_perchannel.dir/depend:
	cd /home/tools/Tengine-tengine-lite/tools/quantize/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tools/Tengine-tengine-lite/tools/quantize /home/tools/Tengine-tengine-lite/tools/quantize /home/tools/Tengine-tengine-lite/tools/quantize/build /home/tools/Tengine-tengine-lite/tools/quantize/build /home/tools/Tengine-tengine-lite/tools/quantize/build/CMakeFiles/quant_tool_uint8_perchannel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/quant_tool_uint8_perchannel.dir/depend

