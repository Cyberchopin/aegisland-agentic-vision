@echo off
setlocal
rem Run from the repository root. No global environment/configuration changes.
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
if not exist work\cuda-build mkdir work\cuda-build
nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 -Xptxas=-v cpp\cuda_gray\gray_bench.cu -o work\cuda-build\gray_bench.exe
exit /b %errorlevel%
