@echo off
rem Build the Strata engine from source (MSVC + CUDA, Ninja) - resumes across runs.
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Dev\StrataHybrid
if not exist build\CMakeCache.txt (
  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89;120 || exit /b 1
)
cmake --build build --target strata -j 6
