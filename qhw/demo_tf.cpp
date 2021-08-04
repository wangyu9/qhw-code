#include "tf_wrapper.h"

int main0() {
	printf("Hello from TensorFlow C library version %s\n", TF_Version());
	return 0;
}


int main() {

	{
		
		auto path = std::string("/media/ssd1/wangyu/WorkSpace/linsolve/apps/");
		auto M = Model(ReadBinaryProto(path + "tf_graphs/y=2x.pb"), { "x" }, "y");

		int64_t dims[1] = { 1l, };
		TF_Tensor* input_tensors[1] = {
			TF_AllocateTensor(TF_FLOAT,dims, 1,sizeof(float))
		};

		static_cast<float*>(TF_TensorData(input_tensors[0]))[0] = 1.f;

		TF_Tensor* out = M(input_tensors, 1);

		float* out_data = static_cast<float*>(TF_TensorData(out));

		printf("Out value %f\n", out_data[0]);

	}

	{

		auto path = std::string("/media/ssd1/wangyu/WorkSpace/linsolve/apps/");
		auto M = Model(ReadBinaryProto(path + "tf_graphs/z=3x+y.pb"), { "x", "y" }, "z");

		int64_t dims[1] = { 1l, };
		TF_Tensor* input_tensors[2] = {
			TF_AllocateTensor(TF_FLOAT,dims, 1,sizeof(float)),
			TF_AllocateTensor(TF_FLOAT,dims, 1,sizeof(float))
		};

		static_cast<float*>(TF_TensorData(input_tensors[0]))[0] = 1.f;
		static_cast<float*>(TF_TensorData(input_tensors[1]))[0] = 2.5f;

		TF_Tensor* out = M(input_tensors, 2);


		float* out_data = static_cast<float*>(TF_TensorData(out));

		printf("Out value %f\n", out_data[0]);

	}


	return 0;
}