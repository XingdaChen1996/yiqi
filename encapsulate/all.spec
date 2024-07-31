# -*- mode: python ; coding: utf-8 -*-


train = Analysis(
    ['D:\\一汽项目_图神经网络\\train.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
train_pyz = PYZ(train.pure)

train_exe = EXE(
    train_pyz,
    train.scripts,
    [],
    exclude_binaries=True,
    name='train',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


Eval = Analysis(
    ['D:\\一汽项目_图神经网络\\eval.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
Eval_pyz = PYZ(Eval.pure)

Eval_exe = EXE(
    Eval_pyz,
    Eval.scripts,
    [],
    exclude_binaries=True,
    name='eval',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

predict = Analysis(
    ['D:\\一汽项目_图神经网络\\predict.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
predict_pyz = PYZ(predict.pure)

predict_exe = EXE(
    predict_pyz,
    predict.scripts,
    [],
    exclude_binaries=True,
    name='predict',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    train_exe,
    train.binaries,
    train.datas,
    Eval_exe,
    Eval.binaries,
    Eval.datas,
    predict_exe,
    predict.binaries,
    predict.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BpNet',
)
