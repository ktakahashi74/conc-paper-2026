# E3 (Consonance-Gated Entrainment) — 認知神経的根拠の検討メモ

## 背景

ALIFE 2026 論文の Experiment 3（internal ID: E5, Fig 5）は、
consonance → metabolic energy → vitality → Kuramoto coupling strength → PLV
という経路（spectral-to-temporal bridge）を検証する pathway-isolation assay。

## 問題

この経路 C → E → v → k_eff → PLV は**モデルに直接配線されている**。
実験結果（C_field と PLV の正相関）は方程式の帰結であり、発見ではなく設計検証（verification）。

より根本的に：**consonance が coupling strength を変調する**という設計選択に
認知神経科学的な根拠がない。DCC（Dynamic Consonance Coupling）として
原理的フレームワークを主張するならば、この橋の justification が必要。

## 現在の論文における記述

- 論文は E3 を "pathway-isolation assay" と位置づけている
- Kuramoto coupling の神経的根拠として Large 1999, Lakatos 2008, Doelling 2015 を引用
- consonance の計算基盤として Plomp-Levelt roughness, harmonicity を引用
- **しかし**、「consonance が coupling strength をスケールする」ことの
  直接的な神経科学的根拠は引用されていない

## 存在しうる根拠の候補

1. **Neural coding efficiency (FFR)**
   - Bidelman & Krishnan (2009): consonant な音程は brainstem FFR で
     より強い phase-locking を示す
   - → より coherent な神経表現 → oscillatory mechanism への drive が強い？
   - 限界：brainstem レベルの知見であり、cortical coupling strength への
     直接的な含意は不明

2. **Predictive coding**
   - consonant = harmonically regular = prediction error が低い
   - → temporal prediction にリソースが割ける → entrainment が強化される？
   - 限界：predictive coding framework からの間接的推論

3. **Salience / attention**
   - consonant な音は処理効率が高い / pleasant
   - → 注意資源の確保 → entrainment の前提条件を満たす
   - 限界：attention → coupling は飛躍がある

4. **Amplitude modulation coherence**
   - consonant な音程（例：3:2）は規則的な AM パターンを生む
   - この AM が cortical entrainment の direct input になりうる
   - → consonance が高い → AM がより periodic → entrainment が強い
   - これが最も direct な経路かもしれない

## 検討すべき選択肢

### A. 認知的根拠を論文に追加
- 上記の候補（特に FFR + AM coherence）を引用
- 「plausible だが未検証の計算仮説」として位置づける
- "energy" = neural coding regularity の proxy、
  "vitality" = oscillatory drive の proxy、と明示的にマッピング

### B. E3 を設計検証として格下げ
- DCC は artistic/engineering design framework であると明記
- この pathway は設計意図の verification であり、認知的必然性は主張しない
- E3 の議論を短縮

### C. E3 を削除/統合
- E2（metabolic selection）と E4（hereditary adaptation）で
  論文の core argument は成立する
- E3 は supplementary に回すか、議論セクションで brief mention
- 9ページ制約下でスペースを他に使える

### D. pathway の再設計
- consonance → coupling の橋を、より神経的に妥当な形に再設計
- 例：AM coherence ベースの entrainment model を採用
- コストが大きいため、この論文のスコープ外か

## 論文全体の argument における E3 の位置

E1: 自己組織化（landscape structure → polyphony）
E2: 代謝的選択（consonance as ecological resource）
**E3: spectral-temporal bridge（consonance → entrainment）**
E4: 遺伝的適応（selection + inheritance）

E3 がなくても E1→E2→E4 の argument は成立するが、
DCC の "temporal dynamics" 軸が欠落する。
E3 を落とすと「consonance は生存と遺伝に影響するが、
リズムとは無関係」という incomplete な picture になる。

## エネルギーモデルの現状（参考）

旧: dE/dt = -c_b + r_E · C_normalized（min-max normalization, binary response）
新: dE/dt = R · max(C_field, 0) - γE （graded steady-state E* = R·C/γ）

パラメータ: R=1.0/s, γ=0.5/s, E_cap=1.0
定常状態: E*(C=0.45) = 0.90, v=0.95; E*(C=0.04) = 0.08, v=0.28
vitality = sqrt(E/E_cap)
coupling: k_eff = K_base · v（vitality condition）
外部 kick: 2 Hz, intrinsic freq: 1.8·2π rad/s, ±2% jitter
