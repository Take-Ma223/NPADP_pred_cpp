動作には C:\PyTorch 直下に
libtorch-win-shared-with-deps-2.0.0+cpu
libtorch-win-shared-with-deps-debug-2.0.0+cpu
が必要です。

exeはprograms直下に配置してください。
programs\model 直下に学習済みモデルmodel.ptを置いてください。

exeファイルを動かすときは以下のdllも必要です。
asmjit.dll
c10.dll
fbgemm.dll
fbjni.dll
libiomp5md.dll
libiompstubs5md.dll
pytorch_jni.dll
torch.dll
torch_cpu.dll
torch_global_deps.dl
uv.dll
