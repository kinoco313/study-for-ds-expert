import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np  # 行列の積などを計算するために使用
    import polars as pl  # 可視化の前処理のために使用。縦持ちデータにするとか

    return np, pl


@app.cell
def _(mo):
    mo.md(
        r"""
    # 主成分分析を理解する

    ## 概要
    1. 複数の変数を1つの変数にする＝縮約するため、各変数に重みをつけたい
    2. 重みをまとめたベクトル＝重みベクトルを考えよう。なぜベクトルを考えるのかというと、個体別の複数の変数をベクトル化したものと重みベクトルの内積が、縮約後の変数＝ **主成分スコア** に該当するから。
    3. 重みに制限をつけないと、主成分スコア（というより、主成分スコアの分散）がいくらでも大きくなり得るため、重みベクトルの大きさが1という制限＝ **制約条件** をつけよう
    4. 重みベクトルの各成分の値を求めるにあたって、主成分スコアの分散が最大となるような重みを求めよう。なぜなら、分散が大きい＝各個体の差異が出ている＝情報量が多いと言えるから。逆に言えば、男性のデータしかないのに、性別という変数を使う意味ないよね？
    5. 重みベクトルの大きさが1になるような制約条件のもと、主成分スコアの分散が最大となるような重みベクトルを求めるためには、 **ラグランジュの未定乗数法** という手法が有効らしい
    6. でも、式変形を進めていくと、**分散共分散行列の固有値と固有ベクトルを求める式** に帰着するから、ラグランジュの未定乗数法を知らなくても重みベクトル＝固有ベクトルを求めること自体はできるよ
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 今回用いる具体例
    - 生徒4人分の数学、物理の点数を用意
    - 2変数程度なら縮約する必要はないが、理解のため簡単な例を利用
    - この2変数を縮約し、「理系力」のような変数を作りたい
    - aさんとcさんが高スコアで、bさんとdさんが低スコアになってほしい
    """
    )
    return


@app.cell
def _(np, pl):
    # 生徒4人分の数学と物理の点数
    student_a = np.array([90, 95])
    student_b = np.array([50, 65])
    student_c = np.array([100, 100])
    student_d = np.array([60, 60])

    # 4行２列の配列を作成(4人分の点数を縦に積み重ねる)
    scores_np = np.vstack([student_a, student_b, student_c, student_d])

    # データフレームも用意しておく
    colmuns = ["math_score", "physics_score"]
    df_score = pl.from_numpy(scores_np, schema=colmuns)
    df_score
    return (df_score,)


@app.cell
def _(df_score, pl):
    # 平均が0だと計算が楽なため、中心化しておく
    df_centered = df_score.with_columns(pl.all() - pl.all().mean())

    df_centered
    return (df_centered,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## これからしたいことをもう少し詳しく説明

    - 変数の個数分の成分を持つ重みベクトル $w=(w_1, w_2)$ を用意。ただし、 $w_1^2+w_2^2=1$ （制約条件）
        - 例えばaさんの主成分スコアは、 $15w_1+15w_2$, dさんの主成分スコアは $-15w_1-20w_2$のように表現できる
    - 主成分スコアの分散が最大になるように、重みベクトル $w$ の成分 $w_1$ と $w_2$ を求める

    ### 前提知識
    - 確率変数 $X_1$, $X_2$に重みをつけて足した$w_1X_1+w_2X_2$の分散は, $w_1^2Var(X_1)+w_1w_2Cov(X_1, X_2)+w_2^2Var(X_2)$ のように、$X_1$ と $X_2$ の分散と共分散をもとに算出できる！
        - ちなみに、$X_1$ と $X_2$ に重みをつけて足す操作（演算）を $X_1$ と $X_2$ の線形結合と呼ぶので、以降はこの言い回しを用いる
    - 行列やベクトルが混じり合うときの積
        - 計算方法というよりかは、入力と出力のパターンをおさえておきたい（理由は後述）
        - 行列とベクトルをかけるとベクトル、ベクトルにベクトルをかけるとスカラーになるって言われてピンとこない場合は、紙とペンで試しに計算してみよう

    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## キーとなる数式
    - $w_1^2Var(X_1)+w_1w_2Cov(X_1, X_2)+w_2^2Var(X_2)$ は確率変数が2つのとき限定なので、確率変数が $n$ 個の場合でも通用するような分散の計算方法が欲しい
    - （結末を知ってるからこそ言えることだが）固有値と固有ベクトルなどが関わってくることから、行列やベクトルを用いて $n$ 個の確率変数を線形結合した場合の分散を求められるようにしたい
    - $n$ 個の確率変数を線形結合を $Z$、重みベクトルを $w=(w_1, w_2, \dots, w_n)$、分散共分散行列を $\Sigma$ とすると求めたい分散は以下のように表現できる

    $$
    Var(Z) = w^T \Sigma w
    $$

    ### 数式展開を追うのが面倒な方へ
    - 実際に数式展開をしなくても、次のようなイメージが湧けばOK
        - とりあえず足し算してる
        - 足されている項は大きく分けると2種類あって、重みの2乗と分散の積（例えば $w_1^2Var(X_1)$ ）と、 $n$ 個の変数から2個の変数を選び、その2変数にかかる重みの積と共分散の積（例えば $w_1w_2Cov(X_1, X_2)$）が足される
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 実際の数値で確かめてみる
    - 数学にかける重みを $0.8$、物理にかける重みを $0.6$ と仮に設定して、重みベクトルと分散共分散行列から算出された値が、定義通り算出される分散に一致するかみてみる
        - $0.8^2+0.6^2=0.64+0.36=1$ となり、これらの重みは制約条件を満たす
        - 本来は、これらの値を最終的に求める
    """
    )
    return


@app.cell
def _(df_centered, pl):
    # 重みづけと、線形結合した結果を算出
    df_weightened = df_centered.with_columns(
        (0.8 * pl.col("math_score")).alias("math_weightened"),
        (0.6 * pl.col("physics_score")).alias("physics_weightened"),
    ).with_columns(
        (pl.col("math_weightened") + pl.col("physics_weightened")).alias("Z")
    )

    df_weightened
    return (df_weightened,)


@app.cell
def _(df_weightened, pl):
    # 標本分散を計算する
    Z_var = df_weightened.select(pl.col("Z").var(ddof=0))
    print(Z_var[0, 0])
    return


@app.cell
def _(df_centered, np):
    # 重みベクトル
    w = np.array([0.8, 0.6])

    # 中心化済みデータを NumPy 配列に
    # df_centered は既に pl.DataFrame として定義済み
    X = df_centered.to_numpy()

    # 標本分散（母分散）としての分散共分散行列 Σ を計算
    # Σ = (X^T X) / n
    n = X.shape[0]
    Sigma = X.T @ X / n

    # 重みベクトルを用いた分散 Var(Z) = w^T Σ w
    variance_Z = w.T @ Sigma @ w

    # 結果を表示
    print(variance_Z)
    return (Sigma,)


@app.cell
def _(Sigma):
    # 分散共分散行列も確認しておく
    print(Sigma)

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 今回用いるデータから、主成分スコアを数式化する
    - 重みベクトル $w=(w_1, w_2)$
    - 分散共分散行列 $\Sigma=\begin{pmatrix}425 & 350\\ 350 & 312.5\end{pmatrix}$
    - 以上を踏まえると、主成分スコアの分散 $f(w_1, w_2)$ は以下のように $w_1$ と $w_2$ の関数となる

    $$
    \begin{align}
    f(w_1, w_2) &= (w_1, w_2)\begin{pmatrix}425 & 350\\ 350 & 312.5\end{pmatrix}\begin{pmatrix}w_1 \\ w_2\end{pmatrix} \\
    &= (w_1, w_2)\begin{pmatrix}425w_1 + 350w_2\\ 350w_1 + 312.5w_2\end{pmatrix} \\
    &= 425w_1^2 + 350w_1w_2 + 350w_2w_1 + 312.5w_2^2 \\
    &= 425w_1^2 + 312.5w_2^2 + 700w_1w_2
    \end{align}
    $$

    この $f(w_1, w_2)$ が $w_1^2 + w_2^2 = 1$ のもとで最大となるような $w_1$ と $w_2$ をもとめるときに、**ラグランジュの未定乗数法** を利用する
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## ラグランジュの未定乗数法について
    - 正直自分もちゃんと理解していない
    - 制約のある関数の最大値をとりうる点を求める方法だと割り切って使う
    - 今回でいえば、 $f(w_1, w_2) = 425w_1^2 + 312.5w_2^2 + 700w_1w_2$ 、 $g(w_1, w_2) = w_1^2 + w_2^2 - 1$ とし、以下のような問題を解く

    $$
    (\frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}) = \lambda(\frac{\partial g}{\partial w_1}, \frac{\partial g}{\partial w_2})
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 計算部分
    - ここからは、 $w_1$, $w_2$, $\lambda$ の3文字を求めるだけなので、偏微分と連立方程式さえわかっていれば解ける
    - 使える式は以下の3つ

    1. $w_1^2 + w_2^2 = 1$
    2. $850w_1 + 700w_2 = 2w_1\lambda$
    3. $625w_2 + 700w_1 = 2w_2\lambda$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    2番と3番は、行列とベクトルを使って以下のように表現できる

    $$
    \begin{align}
    \begin{pmatrix}850 & 700\\700 & 625\end{pmatrix}\begin{pmatrix}w_1 \\ w_2\end{pmatrix} = \lambda\begin{pmatrix}2w_1\\2w_2\end{pmatrix} 
    \end{align}
    $$

    2で括り出すと

    $$
    \begin{align}
    2\begin{pmatrix}425 & 350 \\ 350 & 312.5 \end{pmatrix}\begin{pmatrix}w_1 \\ w_2\end{pmatrix} = 2\lambda\begin{pmatrix}w_1\\w_2\end{pmatrix}
    \end{align}
    $$

    両辺を2で割り、さらに $\Sigma=\begin{pmatrix}425 & 350 \\ 350 & 312.5 \end{pmatrix}$, $w=\begin{pmatrix}w_1 \\ w_2\end{pmatrix}$ であることを踏まえると、上式は以下のように表現できる！（固有値と固有ベクトルを求めるやつじゃん！）

    $$
    \Sigma w = \lambda w
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 頑張って $\lambda$ を求めるパート

    - $\lambda$ に単位行列 $I$ をつけて移項して、重みベクトルはゼロベクトルではないから $det(\Sigma - \lambda I) = 0$ を解けばいいので...
    - 何言ってるかわからないという人は、2次正方行列の固有値と固有ベクトルを求める練習をしてください
    - 分散共分散行列が汚いせいで因数分解するのが面倒そうなので、Pythonに求めてもらおう
    """
    )
    return


@app.cell
def _(Sigma, np):
    # 固有値と固有ベクトルを計算
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

    # --- 結果の解説 ---
    # eigenvalues: 固有値を格納した1次元配列。通常、昇順にソートされています。
    # eigenvectors: 固有ベクトルを「列」として格納した2次元配列（行列）。
    #              i番目の列が、i番目の固有値に対応する固有ベクトルです。

    print("固有値:")
    print(eigenvalues)
    print("\n固有ベクトル:")
    print(eigenvectors)
    return


@app.cell
def _(mo):
    mo.md(
        r"""上記結果から、主成分スコアの分散が最大（およそ723）となるときの重みベクトルは $w=(0.76, 0.65)$ となり（方向さえ合っていればいいので、符号は逆でもいい）、これらの重みを数学と物理の点数にかけて足し合わせることで、「理系力」を導き出すことができる"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 主成分スコアまとめ

    - $n$個の変数に重みをつけて足し合わせることで1つの変数、すなわち主成分スコアに縮約したいとき、$n$個の成分を持つ重みベクトル $w$ と分散共分散行列 $\Sigma$ を用意する
    - 重みをどのようにつけるかというと、主成分スコアの分散が最大となるような重みを求めるようにする。このとき、重みに制約をつけないといくらでも分散は大きくなれるので、重みベクトルの大きさ（各成分の平方和）が1となるようにする
        - 大きさは100でも1億でもいいが、1だと楽なので1にしておく
    - 主成分スコアの分散は重みベクトルと分散共分散行列を使って $w^T \Sigma w$ で算出することができ、 $w_1, w_2\dots w_n$ の関数が得られるため、多変数関数が極値を取るときの点を求めればいい
    - 制約つきの極値を求めるときにはラグランジュの未定乗数法が有効となるが、$\nabla f = \lambda \nabla g$という式を変形すると、 $\Sigma w = \lambda w$ のように、分散共分散行列 $\Sigma$ の固有値と固有ベクトルを求める問題に帰着する
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
