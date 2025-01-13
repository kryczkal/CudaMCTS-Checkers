set(TESTS_EXEC_NAME tests)

############################ Download Google Test ############################
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

############################## Find test files ###############################
file(GLOB_RECURSE TEST_FILES "*.cu" "*.cpp" )

############################# Executable Target ##############################
add_executable(${TESTS_EXEC_NAME} ${TEST_FILES})

################################## Linking ###################################
target_link_libraries(${TESTS_EXEC_NAME} gtest gtest_main)
target_link_libraries(${TESTS_EXEC_NAME} ${PROJECT_LIB_NAME})

################################## Testing ###################################
include(GoogleTest)
gtest_discover_tests(${TESTS_EXEC_NAME})

