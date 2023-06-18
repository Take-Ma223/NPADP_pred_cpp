#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <exception>
#include <direct.h>
#include <stdio.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")

torch::jit::script::Module loadModel();
void startServerLoop();
std::vector<std::string> separate(std::string str);
double getDifficultyValue(std::vector<std::string>);
at::Tensor makeTensor(std::vector<std::string>);
double predict(torch::jit::script::Module& module, at::Tensor tensor);
void makeDirectry(const std::string& dir);
void writeOutput(std::string& text);
void initSocket();

const int ARGC_COUNT = 24;

const std::string saveDataPath = "save_data/";
const std::string tmpPath = "tmp/";
const std::string apdPath = "auto_difficulty_prediction/";
const std::string savePath = saveDataPath + tmpPath + apdPath;
const std::string saveFileName = "pop.txt";

torch::jit::script::Module module;

static const int BUFFER_SIZE = 256;
/* �|�[�g�ԍ��A�\�P�b�g */
static const unsigned short port = 50001;
int srcSocket;  // ����
int dstSocket;  // ����

/* sockaddr_in �\���� */
struct sockaddr_in srcAddr;
struct sockaddr_in dstAddr;
int dstAddrSize = sizeof(dstAddr);

int main() {
	try {
		// torch::jit::script::Module �^�� module �ϐ��̒�`
		module = loadModel();
	}
	catch (c10::Error e) {
		writeOutput(std::string(e.what()));
		exit(-1);
	}

	initSocket();

	startServerLoop();

	//�I��
	exit(0);
}

void startServerLoop()
{
	int numrcv;

	char buffer[BUFFER_SIZE];
	int err = 0;

	/* �ڑ��̎�t�� */
	printf("Waiting for connection ...\n");
	dstSocket = accept(srcSocket, (struct sockaddr*)&dstAddr, &dstAddrSize);
	err = WSAGetLastError();
	printf("Connected from %s\n", inet_ntoa(dstAddr.sin_addr));

	/* �p�P�b�g��M */
	while (1) {
		//��M
		numrcv = recv(dstSocket, buffer, BUFFER_SIZE, 0);
		err = WSAGetLastError();
		if (numrcv == 0 || numrcv == -1) {
			int status = closesocket(dstSocket);
			break;
		}
		printf("received: %s\n", buffer);

		auto strList = separate(buffer);
		if (strList.size() != ARGC_COUNT) {
			send(dstSocket, "error", strnlen_s("error", 100), 0);
		}
		else {
			double difficultyValue = getDifficultyValue(strList);
			auto result = std::to_string(difficultyValue);
			send(dstSocket, result.c_str(), strnlen_s(result.c_str(), 100), 0);
		}
	}
	closesocket(dstSocket);
}

/// <summary>
/// ��������J���}�ŋ�؂��Ĕz��Ɋi�[
/// </summary>
/// <param name="str">�J���}��؂�̕�����</param>
/// <returns>������z��</returns>
std::vector<std::string> separate(std::string str) {
	auto separator = std::string(",");         // ��؂蕶��
	auto separator_length = separator.length(); // ��؂蕶���̒���
	auto list = std::vector<std::string>();
	if (separator_length == 0) {
		list.push_back(str);
	}
	else {
		auto offset = std::string::size_type(0);
		while (1) {
			auto pos = str.find(separator, offset);
			if (pos == std::string::npos) {
				list.push_back(str.substr(offset));
				break;
			}
			list.push_back(str.substr(offset, pos - offset));
			offset = pos + separator_length;
		}
	}

	return list;
}

double getDifficultyValue(std::vector<std::string> strList)
{
	auto tensor = makeTensor(strList);
	double result = predict(module, tensor);
	result = round(result * 100) / 100;

	//std::ostringstream oss;
	//oss << std::fixed;
	//oss << std::setprecision(2) << result;

	//return oss.str();
	////���ʏ�������
	//writeOutput(oss.str());

	return result;
}


torch::jit::script::Module loadModel()
{
	torch::jit::script::Module module;

	// �w�K�ς݃��f���̓ǂݍ���
	module = torch::jit::load("programs/application/auto_difficulty_prediction/model/model.pt", torch::kCPU);

	return module;
}

at::Tensor makeTensor(std::vector<std::string> strList)
{
	// ���f���ւ̓��̓e���\��
	auto tensor = torch::randn({ 1, ARGC_COUNT }).to("cpu");

	for (int i = 0; i < ARGC_COUNT; i++) {
		int argvInd = i;
		tensor[0][i] = atoi(strList[argvInd].c_str());
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

void writeOutput(std::string& text)
{
	makeDirectry(saveDataPath);
	makeDirectry(saveDataPath + tmpPath);
	makeDirectry(saveDataPath + tmpPath + apdPath);


	std::ofstream writing_file;
	writing_file.open(savePath + saveFileName, std::ios::out);
	writing_file << text;
	writing_file.close();
}

void makeDirectry(const std::string& dir) {
	//�f�B���N�g���쐬
	_mkdir(dir.c_str());
}

/// <summary>
/// �\�P�b�g������
/// </summary>
void initSocket() {
	/* �e��p�����[�^ */
	int status;


	/************************************************************/
	/* Windows �Ǝ��̐ݒ� */
	WSADATA data;
	WSAStartup(MAKEWORD(2, 0), &data);

	/* sockaddr_in �\���̂̃Z�b�g */
	memset(&srcAddr, 0, sizeof(srcAddr));
	srcAddr.sin_port = htons(port);
	srcAddr.sin_family = AF_INET;
	srcAddr.sin_addr.s_addr = htonl(INADDR_ANY);

	/* �\�P�b�g�̐��� */
	srcSocket = socket(AF_INET, SOCK_STREAM, 0);

	/* �\�P�b�g�̃o�C���h */
	auto bindResult = bind(srcSocket, (struct sockaddr*)&srcAddr, sizeof(srcAddr));
	int err = WSAGetLastError();

	/* �ڑ��̋��� */
	listen(srcSocket, 1);
	err = WSAGetLastError();

	/* Windows �Ǝ��̐ݒ� */
	//WSACleanup();
}