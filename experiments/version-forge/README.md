# version-forge — バージョン偽造・配置偽造・Force-Store・エラー外積DP

NumPy のバージョンに厳しいライブラリは TexNP（依存ゼロの生成バイナリ）で
動くか、という問いから始まった実験群（2026-10-30、この VM で実測）。
相手は NumPy 1.26.4 向けにビルドされた拡張が NumPy 2.2.6 で死ぬ現象で、
代表例は pandas 2.1.4 / scipy 1.11.4 の import 失敗
（`numpy.dtype size changed` / `numpy.core.multiarray failed to import`）。

## 結論の地図

関門は2種類あり、運命が分かれる。

| 関門 | 方式 | 結果 |
|---|---|---|
| バージョン文字列 | 文字列偽造（トライエラー） | **通る**。pandas は実体 1.26 のまま "2.2.6" 偽称で計算まで正常 |
| バイナリ配置（ABI） | バイナリパッチ | 検査は開くが消費者は旧オフセットで読み続け**黙って壊れる** |
| バイナリ配置（ABI） | Force-Store（load 先へ store） | 検査・消費者とも無改造で通る。ただし実体単位・揮発・副作用あり |
| 形式そのもの | エラー外積 DP | 値を覚えず**規則を推定**し、未見の型族にも一般化 |

## ファイル

- `version_forge.py` + `strict_lib.py` — バージョン文字列の偽造。
  実 numpy の辞書を複写し `__version__` だけ書き換えた shim を差し込み、
  候補を新品サブプロセスで試し続ける。文字列関門は通る。ABI 関門
  （実 numpy2 上の scipy）は候補4件全滅し、全滅自体が診断になる。
- `gatelib.c` — numpy1 前提の最小 C 拡張。`check` = descr サイズ検査
  （pandas の .so に埋め込まれた検査と同型）、`consume` = 旧配置
  （offset 32 の int32）で elsize を読む消費者。期待値定数はマーカー
  文字列 `FORGE96!` の直後に置き、バイナリパッチ可能にしてある。
  ビルド: `gcc -shared -fPIC -O2 $(python3-config --includes) gatelib.c -o gatelib.so`
- `force_store.py` — 検証部への強制入力。checker が load するスロットへ
  ctypes で store し、総当たり試行（サブプロセス隔離、segfault を吸収）
  でパスまで繰り返す。検査は1試行、消費者は9試行（うち segfault 2回）。
- `force_multiplex.py` — インスタンス多重化。組み込み dtype は numpy の
  キャッシュが同一実体を返すため、全18種を先行生成して全部に store
  すれば 16 種は閉じる。可変長（bytes_/str_）と事後生成の型族は逃げる。
  itemsize=1 の3種では numpy 自身が内部エラーのトレースバックを吐く
  （副作用の実発火）。
- `error_dp.py` — エラーを包み込む動的プログラム。Stage A: 全スロットに
  自分のオフセット値をマーカーとして store し、消費者の誤答（エコー）
  から読み出し位置 32 と幅 4B を同定（探査2回、クラッシュ0）。Stage B:
  誤差行列 E = read − truth の外積蓄積 G = EᵀE でフィールドを分類
  （slot40 = elsize は誤差消滅、slot48 = alignment は誤差の非零が
  複素数型2件分 16+64=80 と一致）。Stage C: 出力は翻訳規則
  `store int32@32 ← load int64@40`。多重化から逃亡した構造化・
  datetime・可変長にも未見で適用でき、全件真値と一致。
- `wrap_fix.py` — slot32 衝突の修正＝時分割ラップ。保存 → store →
  呼出 → 復元（try/finally）。全5件で真値一致、副作用トレースバック
  完全消失。前提は同期的消費。
- `thread_scope.py` — 時分解の置き場所＝スレッドスコープの検証。
  同一実体へ 8 スレッド×3000 回同時実行すると誤 consume（0〜2件）と
  偽値の残留が発生（試行ごとに変動、残留は再現性あり）。ロック所有
  (L) と twin 実体分割 (T) では誤り 0、T のみ共有実体が完全無傷。
- `error-corpus/` — 意図的にエラーを出す Python スクリプト14本と
  収穫係 `collect_errors.py`（→ `errors.json`）。例外11種＋シグナル死＋
  無言の終了コード＋正常対照。エラーを捨てず標本として集める側の道具。

## 再現に必要なもの

- 実験環境: Python 3.12、numpy 1.26.4（実体）＋ numpy 2.2.6 の重ね置き
  （`pip install --target=/tmp/np2 --no-deps numpy==2.2.6` し、
  `PYTHONPATH=/tmp/np2` で切り替える）。
- スクリプトは相対パス化済み。`gatelib.so` は上記 gcc でビルドする
  （.so はリポジトリに含めない）。

## 正直な境界

- pandas/scipy「そのもの」は TexNP では動かない（Python C 拡張が
  numpy C-API を import する構造）。動くのは計算の書き直し。
- 配置の実体（フィールドオフセット）はランタイムの所有物で、
  拡張側からは偽造できない。できたのは検査の偽造と、load 先への
  store と、形式の推定と、時分割による衝突回避まで。
- 時分割はスレッドスコープ内でのみ正しい。協調しない外部読み手への
  保証はない。
