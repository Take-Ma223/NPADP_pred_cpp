#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <exception>
#include <direct.h>

using namespace std;

torch::jit::script::Module loadModel();
at::Tensor makeTensor(const char* argv[]);
double predict(torch::jit::script::Module& module, at::Tensor tensor);
void makeDirectry(const string& dir);
void writeOutput(string& text);

const int ARGC_COUNT = 25;

const string saveDataPath = "save_data/";
const string tmpPath = "tmp/";
const string apdPath = "auto_difficulty_prediction/";
const string savePath = saveDataPath + tmpPath + apdPath;
const string saveFileName = "pop.txt";


int main(int argc, const char* argv[]) {
	if (argc != ARGC_COUNT) {
		writeOutput(string("�G���[ �����̐����s���ł�"));
		exit(-1);
		//�����̐����������Ȃ�
	}

	torch::jit::script::Module module;
	try {
		// torch::jit::script::Module �^�� module �ϐ��̒�`
		module = loadModel();
	}
	catch (std::ios_base::failure e) {
		writeOutput(string(e.what()));
		exit(-1);
	}

	auto tensor = makeTensor(argv);
	double result = predict(module, tensor);

	//���ʏ�������
	writeOutput(to_string(result));

	//�I��
	exit(0);
}

torch::jit::script::Module loadModel()
{
	torch::jit::script::Module module;

	// �w�K�ς݃��f���̓ǂݍ���
	try {
		module = torch::jit::load("programs/application/auto_difficulty_prediction/model/model.pt", torch::kCPU);
	}
	catch (const c10::Error& e) {
		throw std::ios_base::failure("�G���[ model.pt�t�@�C����������܂���B");
	}

	return module;
}

at::Tensor makeTensor(const char* argv[])
{
	// ���f���ւ̓��̓e���\��
	auto tensor = torch::randn({ 1, 24 }).to("cpu");

	for (int i = 0; i < 24; i++) {
		int argvInd = i + 1;
		tensor[0][i] = atoi(argv[argvInd]);
	}

	return tensor;
}

double predict(torch::jit::script::Module& module, at::Tensor tensor)
{
	std::vector<torch::jit::IValue> input;
	input.push_back(tensor);

	// ���_
	auto res = module.forward(input);

	//�o�̓e���\����Double�^�ɕϊ�
	auto output = res.toTensor().item().toDouble();
	return output;
}

void writeOutput(string& text)
{
	makeDirectry(saveDataPath);
	makeDirectry(saveDataPath + tmpPath);
	makeDirectry(saveDataPath + tmpPath + apdPath);


	ofstream writing_file;
	writing_file.open(savePath + saveFileName, ios::out);
	writing_file << text;
	writing_file.close();
}

void makeDirectry(const string& dir) {
	//�f�B���N�g���쐬
	_mkdir(dir.c_str());
}