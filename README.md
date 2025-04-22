# mem-rearrange

[![CI](https://github.com/InfiniTensor/mem-rearrange/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/InfiniTensor/mem-rearrange/actions)
[![Latest version](https://img.shields.io/crates/v/mem-rearrange.svg)](https://crates.io/crates/mem-rearrange)
[![Documentation](https://docs.rs/mem-rearrange/badge.svg)](https://docs.rs/mem-rearrange)
[![license](https://img.shields.io/github/license/InfiniTensor/mem-rearrange)](https://mit-license.org/)
[![codecov](https://codecov.io/github/InfiniTensor/mem-rearrange/branch/main/graph/badge.svg)](https://codecov.io/github/InfiniTensor/mem-rearrange)

![GitHub repo size](https://img.shields.io/github/repo-size/InfiniTensor/mem-rearrange)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/InfiniTensor/mem-rearrange)
[![GitHub Issues](https://img.shields.io/github/issues/InfiniTensor/mem-rearrange)](https://github.com/InfiniTensor/mem-rearrange/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/mem-rearrange)](https://github.com/InfiniTensor/mem-rearrange/pulls)
![GitHub contributors](https://img.shields.io/github/contributors/InfiniTensor/mem-rearrange)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/InfiniTensor/mem-rearrange)

根据布局参数重排数据，将数据从一个高维数组中转移到另一个具有不同存储布局的高维数组中。

不妨将数据来源的数组称为 `src`，去向的数组称为 `dst`。由于两个数组形状相同，二者的元素是一一对应的。算子将 `src` 中的数据拷贝到 `dst` 中的对应位置。

例如，要转置存放在数组中的 $2 \times 3$ 矩阵：

> - 符号表示数据元素；
> - 下标表示数据在存储空间中排布的位置；

$$
src:
\left(
    \begin{gathered}
    a_0, b_1, c_2\\
    d_3, e_4, f_5\\
    \end{gathered}
\right)
\stackrel{转置}{\rightarrow}
dst:
\left(
    \begin{gathered}
    a_0, d_1\\
    b_2, e_3\\
    c_4, f_5\\
    \end{gathered}
\right)
$$

这个操作实际改变了存储空间中的数据排布，因此必须进行数据重排。下式直观展示了数据重排的功能：

> 注意观察表示数据排布的下角标。

$$
src:
\left(
    \begin{gathered}
    a_0, b_1, c_2\\
    d_3, e_4, f_5\\
    \end{gathered}
\right)
\stackrel{布局转置}{\rightarrow}
\left(
    \begin{gathered}
    a_0, d_3\\
    b_1, e_4\\
    c_2, f_5\\
    \end{gathered}
\right)
\stackrel{数据重排}{\rightarrow}
dst:
\left(
    \begin{gathered}
    a_0, d_1\\
    b_2, e_3\\
    c_4, f_5\\
    \end{gathered}
\right)
$$

由于重排操作不关心数据的内容，仅在存储空间上操作数据，可以对算法进行硬件无关的预排布和优化。

优化过程包括 3 个主要步骤：

1. 剔除长度为 1 的维度；
2. 按步长对维度排序；
3. 合并在存储排布上连续的维度；

完成以上步骤后，关于重排运算的全部信息将存储在 `Rearranging` 结构体中。通过为结构体传入不同的数据指针可以反复执行相同的重排变换。CPU 的重排运算实现使用 rayon 进行并行加速。
