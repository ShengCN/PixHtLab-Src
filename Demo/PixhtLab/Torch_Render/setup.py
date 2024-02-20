from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
	name='hshadow',
	ext_modules=[
		CUDAExtension('hshadow', [
			'hshadow_cuda.cpp',
			'hshadow_cuda_kernel.cu',
		])
	],
	cmdclass={
		'build_ext': BuildExtension
	}
)

# setup(
# 	name='plane_visualize',
# 	ext_modules=[
# 		CUDAExtension('plane_visualize', [
# 			'plane_visualize.cpp',
# 			'plane_visualize_cuda.cu',
# 		])
# 	],
# 	cmdclass={
# 		'build_ext': BuildExtension
# 	}
# )