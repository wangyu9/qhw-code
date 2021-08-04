
#include <tensorflow/c/c_api.h>
#include <string>
#include <vector>

/**
 * Load a protobuf buffer from disk,
 * recreate the tensorflow graph and
 * provide it for inference.
 */

class Model {

private:
	TF_Graph* graph;
	TF_Session* session;

	std::vector<TF_Operation*> input_ops;
	TF_Output* inputs;

	TF_Operation* output_op;
	TF_Output 		output;



	/**
	 * Tensorflow does not throw errors but manages runtime information
	 *   in a _Status_ object containing error codes and a failure message.
	 *
	 * AssertOk throws a runtime_error if Tensorflow communicates an
	 *   exceptional status.
	 *
	 */
	void AssertOk(const TF_Status* status) const;

public:
	/**
	 * binary_graphdef_protobuf_filename: only binary protobuffers
	 *   seem to be supported via the tensorflow C api.
	 * input_node_name: the name of the node that should be feed with the
	 *   input tensor
	 * output_node_name: the node from which the output tensor should be
	 *   retrieved
	 */
	Model(//const std::string& binary_graphdef_protobuf_filename,
		TF_Buffer* graph_def,
		const std::vector<std::string>& input_node_names,
		const std::string& output_node_name);

	/**
	 * Clean up all pointer-members using the dedicated tensorflor api functions
	 */
	~Model();

	/**
	 * Run the graph on some input data.
	 *
	 * Provide the input and output tensor.
	 */
	TF_Tensor* operator()(TF_Tensor* const* input_tensors, int ninputs) const;

};

TF_Buffer* ReadBinaryProto(const std::string& fname);