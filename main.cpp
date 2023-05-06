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
		writeOutput(string("エラー 引数の数が不正です"));
		exit(-1);
		//引数の数が正しくない
	}

	torch::jit::script::Module module;
	try {
		// torch::jit::script::Module 型で module 変数の定義
		module = loadModel();
	}
	catch (std::ios_base::failure e) {
		writeOutput(string(e.what()));
		exit(-1);
	}

	auto tensor = makeTensor(argv);
	double result = predict(module, tensor);

	//結果書き込み
	writeOutput(to_string(result));

	//終了
	exit(0);
}

torch::jit::script::Module loadModel()
{
	torch::jit::script::Module module;

	// 学習済みモデルの読み込み
	try {
		module = torch::jit::load("programs/application/auto_difficulty_prediction/model/model.pt", torch::kCPU);
	}
	catch (const c10::Error& e) {
		throw std::ios_base::failure("エラー model.ptファイルが見つかりません。");
	}

	return module;
}

at::Tensor makeTensor(const char* argv[])
{
	// モデルへの入力テンソル
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

	// 推論
	auto res = module.forward(input);

	//出力テンソルをDouble型に変換
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
	//ディレクトリ作成
	_mkdir(dir.c_str());
}