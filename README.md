# Parallel Programming — Image and Video Processing

C++/OpenCV coursework comparing serial image-processing routines with SIMD implementations.

## Exercises

- `Q1/`: image/logo blending with serial and SIMD routines.
- `Q4/`: video motion detection with serial and SIMD routines.
- `assets/`: original exercise images and video.
- `PP-CA1-Fall03.pdf`: assignment description.

## Requirements and build

The CMake files request CMake 3.29 and C++23. Use OpenCV and an x86 toolchain compatible with the intrinsics used by the exercises; Q4 also includes MSVC's `intrin.h`.

The original source contains developer-specific OpenCV and asset paths. Update `CMakeLists.txt` and the input paths in each `.cpp` file before building:

```sh
cmake -S Q1 -B build/Q1
cmake --build build/Q1
```

Repeat with `Q4` to build the motion-detection exercise.

The programs display results in OpenCV windows and print timing comparisons. No new speedup measurements or correctness claims are made by this documentation update.

## Credits and source

This is a public portfolio copy of the shared project [MMDSour/PP_CA1](https://github.com/MMDSour/PP_CA1), prepared for Sobhan Alaeddini's account. Original authorship and existing license notices are retained. See [SOURCE.md](SOURCE.md) for the source revision and publication notes.
