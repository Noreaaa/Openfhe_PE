# PFHE 引用论文的安全参数核查（2026-10-23）

核查对象：用户提供的 `/root/PFHE_ASPDAC_2027.pdf`，7 页、25 条参考文献。重点为表 3 的三个比较对象，以及直接相关的 CNN/编译器工作。本文区分作者报告的安全估计、公开代码配置和独立验证；没有重新运行这些论文全部参数的 lattice estimator，也没有复现其完整网络，因此“作者报告达到”不等于本次独立认证。

## 表 3 的直接比较对象

| PFHE 引用 | 论文 | 结论及证据 |
|---|---|---|
| [12] | Orion, ASPLOS 2025 | 有明确 128-bit 安全设计依据。§2.1/2.4 讨论 N、Q 和为保持 128-bit 而调整 dnum；§8.1 给出降低电路深度后仍达 128-bit 的实例。论文使用 Lattigo v5.0.2；不能用我们 OpenFHE 的三位 HYBRID 配置直接裁定其参数。 |
| [20] | SHE, NeurIPS 2019 | 正式发表版 §4（PDF 第 5–6 页）明确报告整体安全估计 λ=152，三个层级分别报告 194/152/289。**但作者公开样例 `src/SHE.cpp:89` 设置 `minimum_lambda=80`，随后调用 TFHE 默认参数构造器。** 这说明论文的三级 leveled-TFHE 配置与公开门级示例不能直接等同。不能断言论文只有 80-bit，也不能把该样例当作复现了论文的 152-bit。 |
| [24] | DCT-CryptoNets, ICLR 2025 | 正文和附录中未找到明确的 128-bit 数字或逐模型安全估计表。作者源码使用 Concrete-ML 默认编译路径，当前公开环境列出 Concrete-ML 1.6.1、Concrete-Python 2.7.0，配置仅设显示选项，没有显式降低安全级别。Concrete 官方文档说明后端按固定安全目标选参，典型为 128-bit；这是**间接支持**，尚不足以独立确认论文每项实验参数。不能判定其低于 128-bit。 |

来源：

- [Orion 正文](https://arxiv.org/html/2311.03470v3)，§2、§7、§8.1、表 2。
- [SHE 正式论文](https://papers.neurips.cc/paper/9194-she-a-fast-and-accurate-deep-neural-network-for-encrypted-data.pdf)，§4；[作者样例固定提交](https://github.com/qianlou/SHE/blob/38642edda8bc1fbb468c034f8ea8d3f110dcfe3a/src/SHE.cpp#L89)。PFHE 的参考文献写的是 arXiv:1906.00148；该预印本与正式版内容不同，引用时应明确版本。
- [DCT 正文](https://arxiv.org/html/2408.15231v2)，§4.1、附录 A.7；[作者编译代码](https://github.com/ar-roy/dct-cryptonets/blob/13e1ca7061e41610dd1574a6a68a657be43db0ad/dct-cryptonets/homomorphic_eval.py#L257)；[作者环境文件](https://github.com/ar-roy/dct-cryptonets/blob/13e1ca7061e41610dd1574a6a68a657be43db0ad/env.yml)；[Concrete 参数优化说明](https://docs.zama.org/concrete/2.4/explanations/compilation/optimizer)。后者是 2.4 文档，对论文精确后端版本和导出参数的复核仍有缺口。

DCT 附录采用 `PBS error probability=0.01`，指单次 PBS 的正确性/失败概率，**不是攻击成功概率或 128-bit 的替代指标**。降低量化位数也不意味着将密码安全等级降为该位数。具体密码参数由后端根据精度、错误概率和安全目标联合选择。

## 其他直接相关的引用

| 引用 | 工作 | 当前核查结论 |
|---|---|---|
| [3] | TenSEAL | 论文 §5 明确以 128-bit 为目标；CNN 使用 N=8192、系数模数 206 bit、scale=21 bit。是降低数值精度而保留密码安全的实例。 |
| [9] | EVA | §8.1 明确所有实验采用 SEAL 默认 128-bit 安全级别。 |
| [18] | Fhelipe | §7.2 明确默认 128-bit 自动选参，带 bootstrap 的实验使用 n=32K 个 slots、报告模数位宽 1552；文中还单列 bootstrap 和密钥切换开销。**这里 n 是 slots，不是 OpenFHE 的 ringDim**，不能把 32K 直接代入我们的环维数表。 |
| [19] | Lee 等，arXiv:2106.07229 | **所引用的 2021 年版本明确是 98-bit，不是 128-bit。** §3.2.1/表 2：N=2^16、秘密汉明重量 64、报告最大模数 1750 bit；§5 明确讨论提高安全性会增加运行时间。这里只评价此版本，不扩展到后续论文。 |
| [21] | Falcon | §4 的 Cryptosystems Settings 明确报告超过 128-bit，并说明用 lwe_estimator 检查。采用 BFV，N=16384、报告系数模数约 440 bit；不是我们当前 CKKS HYBRID 配置。 |
| [10] | CHET | 已读取所引 arXiv 正文，§6.2 描述从 Q 选择足够大的 N，但未从该版本找到明确 128-bit 实验数值；不能借用 EVA 后来的默认配置替其补证。 |
| [15] | CryptoNets | 已读取 ICML 正文，未找到足以给全部实验贴上 128-bit 标签的明确数值声明。也不应将 CryptoNets 说成 CKKS 实现：其发表早于 2017 年 CKKS。 |
| [23] | Cheetah（HPCA 2021） | 已读原论文；这是 BFV 与客户端交互的设计，安全模型和我们的非交互方案不同。当前未确认明确的 128-bit 实验声明，不能直接归类。不要与其他同名 Cheetah 工作混淆。 |
| [17] | Kim & Guyot, TIFS 2023 | 获取了作者实现和其 Lattigo 分支的 bootstrap 参数；尚未获得论文完整安全估计段落。代码使用 N=2^16 的自定义参数集，不能仅凭“用了 Lattigo”认证 128-bit。结论保留。 |
| [25] | CHE, IEEE Access 2022 | 在公开论文全文的 §V.B Experimental Setup 确认作者明确报告全部 PPDL 实验达到 128-bit；使用 Microsoft SEAL 3.7 / Python-SEAL、RNS-CKKS。该证据为论文公开全文，不是引用它的其他论文。尚未独立重跑估计。 |
| [16] | GAZELLE | §7.1 明确两部分参数均按 128-bit 选择。PAHE 使用 BFV，n=2048、密文模数 60 bit、明文模数 20 bit、σ=4；非线性使用交互式 garbled circuits。因此，小环维数下达到 128-bit 与需要长模数链的非交互 CKKS 是不同问题。 |
| [5] | Chabanne 等，2017 | 本次未完成其数值安全参数核查，保留。 |

一手来源：

- [TenSEAL PDF](https://arxiv.org/pdf/2104.03152)，§5，PDF 第 5 页附近。
- [EVA PDF](https://arxiv.org/pdf/1912.11951)，§8.1，PDF 第 12 页。
- [Fhelipe 论文](https://doi.org/10.1145/3656382)，§7.2；[作者论文在 NSF 的存档](https://par.nsf.gov/servlets/purl/10523620)。
- [Lee 等的所引预印本](https://arxiv.org/pdf/2106.07229)，§3.2.1/表 2、§5，PDF 第 5、9 页。
- [Falcon PDF](https://papers.nips.cc/paper/2020/file/18fc72d8b8aba03a4d84f66efabce82e-Paper.pdf)，§4，PDF 第 7 页。
- [CHET PDF](https://arxiv.org/pdf/1810.00845)，§6.2。
- [CryptoNets PDF](https://proceedings.mlr.press/v48/gilad-bachrach16.pdf)。
- [Cheetah 原论文](https://arxiv.org/pdf/2006.00505)，§II/III。
- [Kim & Guyot 作者代码](https://github.com/dwkim606/optimal_conv)，[其 Lattigo 分支](https://github.com/dwkim606/test_lattigo)。
- [CHE DOI](https://doi.org/10.1109/ACCESS.2022.3210134)、[论文公开全文](https://www.researchgate.net/publication/363865180_CHE_Channel-wise_Homomorphic_Encryption_for_Ciphertext_Inference_in_Convolutional_Neural_Network)（§V.B），[GAZELLE 正文](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-juvekar.pdf)（§7.1），[Chabanne 等论文](https://eprint.iacr.org/2017/035)。

其余参考文献 [1][2][6] 为库，[4][7][8][11][14] 为基础密码方案/学位论文，[13] 为库测试，[22] 为特征反演研究。引用一个方案或库，不代表选用该库任意配置的实验自动有 128-bit 安全；它们也不是可直接拿来给 CNN 耗时贴统一安全标签的模型实验。

## 对 PFHE 草稿的直接影响

1. PDF 没有明确写出“128-bit”，但摘要和结论有保持保密性/安全性的表述。§5.2 和表 1 使用 N=2^11；文中没有足以复核安全性的完整 Q/P、秘密分布、FHEW 参数和攻击估计。不能从这些结果支持整体 128-bit。
2. 当前代码的 `HEStd_NotSet` 与 `TOY` 进一步说明，现有结果属于研究测试参数。它们的速度/内存优势不能标作与上述 128-bit 实验同等安全条件下的优势。应在比较表新增“安全配置/估计方法、实测或估算、线程/硬件、端到端或前缀”等列。
3. §3.1 写 FHEW keys 发送给服务端，需要明确限定为评估/转换密钥；**不能把 FHEW 私钥也交给能访问 CKKS→FHEW 转换结果的服务端**。旧解密符号实现与新加密回转实现要分别描述。位安全等级不能弥补私钥暴露。
4. 表 3 的原始时间还包含不同线程和方法。DCT 16115 秒是 RGB 输入的 ImageNet 行，不是其频域最优 8562 秒行；Orion 的 14338 秒对应单线程 ResNet-34/SiLU 和完整网络；SHE 216000 秒须核对测量/估算及其三级 LHE 配置。不能仅凭总耗时宣称同任务、同安全、同资源的速度优势。
5. 可采用的临时措辞：当前 PFHE 数据为“experimental parameters without an established 128-bit security claim”；同文献时间的比较为“indicative comparison across different security, hardware, and evaluation settings”。若要宣称标准 128-bit，应重新选参、明确整个 CKKS/FHEW 密钥接口与威胁模型，并重跑相关对照。

本次没有修改用户原 PDF，也没有将缺乏证据等同于论文不安全。原论文/公开代码的摘录文本和下载文件保存在 `/root/paper_security_audit/`。
