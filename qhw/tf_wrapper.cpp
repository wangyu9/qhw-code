#include "tf_wrapper.h"

#include <stdio.h>
#include <tensorflow/c/c_api_experimental.h>

// modified from https://github.com/adriankoering/tensorflow-cc-inference/blob/master/lib/Inference.cc

#include <fstream>
#include <sstream>
#include <exception>

/**
 * Read a binary protobuf (.pb) buffer into a TF_Buffer object.
 *
 * Non-binary protobuffers are not supported by the C api.
 * The caller is responsible for freeing the returned TF_Buffer.
 */
TF_Buffer* ReadBinaryProto(const std::string& fname)
{
	std::ostringstream content;
	std::ifstream in(fname, std::ios::in | std::ios::binary); // | std::ios::binary ?

	if (!in.is_open())
	{
		throw std::runtime_error("Unable to open file: " + std::string(fname));
	}

	// convert the whole filebuffer into a string
	content << in.rdbuf();
	std::string data = content.str();

	return TF_NewBufferFromString(data.c_str(), data.length());

	return NULL; // wangyu removed this fun.
}

/**
 * Tensorflow does not throw errors but manages runtime information
 *   in a _Status_ object containing error codes and a failure message.
 *
 * AssertOk throws a runtime_error if Tensorflow communicates an
 *   exceptional status.
 *
 */
void Model::AssertOk(const TF_Status* status) const
{
	if (TF_GetCode(status) != TF_OK)
	{
		throw std::runtime_error(TF_Message(status));
	}
}

/**
 * Load a protobuf buffer from disk,
 * recreate the tensorflow graph and
 * provide it for inference.
 */
Model::Model(
	//	const std::string& binary_graphdef_protobuffer_filename,
	TF_Buffer* graph_def,
	const std::vector<std::string>& input_node_names,
	const std::string& output_node_name)
{
	// init the 'trival' members
	TF_Status* status = TF_NewStatus();
	graph = TF_NewGraph();

	// create a bunch of objects we need to init graph and session
	// TF_Buffer* graph_def = ReadBinaryProto(binary_graphdef_protobuffer_filename);
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	TF_SessionOptions* session_opts = TF_NewSessionOptions();

	// import graph
	TF_GraphImportGraphDef(graph, graph_def, opts, status);
	AssertOk(status);

	// and create session
	session = TF_NewSession(graph, session_opts, status);
	AssertOk(status);

	// prepare the constants for inference
	// input

	inputs = new TF_Output[input_node_names.size()];

	int i = 0;
	for (auto pname = input_node_names.begin(); pname != input_node_names.end(); ++pname) {
		auto op = TF_GraphOperationByName(graph, pname->c_str());
		input_ops.push_back(op);
		inputs[i] = { op, 0 };
		i++;
	}

	// output
	output_op = TF_GraphOperationByName(graph, output_node_name.c_str());
	output = { output_op, 0 };

	// Clean Up all temporary objects
	TF_DeleteBuffer(graph_def);
	TF_DeleteImportGraphDefOptions(opts);
	TF_DeleteSessionOptions(session_opts);

	TF_DeleteStatus(status);
}

Model::~Model()
{
	TF_Status* status = TF_NewStatus();
	// Clean up all the members
	TF_CloseSession(session, status);
	TF_DeleteGraph(graph);
	//TF_DeleteSession(session); // TODO: delete session?

	TF_DeleteStatus(status);
	// input_op & output_op are delete by deleting the graph
}

TF_Tensor* Model::operator()(TF_Tensor* const* input_tensors, int ninputs) const
{
	TF_Status* status = TF_NewStatus();
	TF_Tensor* output_tensor;
	TF_SessionRun(session, nullptr,
		inputs, input_tensors, ninputs,
		&output, &output_tensor, 1,
		&output_op, 1,
		nullptr, status);
	AssertOk(status);
	TF_DeleteStatus(status);

	return output_tensor;
}

