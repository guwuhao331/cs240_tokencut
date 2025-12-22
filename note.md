# Normalized Cut (Ncut) 与 Fast Normalized Cut (FastNCut) 原理笔记

## 1. Normalized Cut (Ncut): 从“最小割”到“归一化割”

### (1) 为什么原始的 Min-Cut 不行？
在图论中，图像分割被看作是 **图划分 (Graph Partitioning)** 问题。
* **节点 (Nodes)**：图像中的像素（在 TokenCut 项目中是 Transformer 的 Patch Token）。
* **边 (Edges)**：节点之间的连接，权重 $w_{ij}$ 代表两个节点的**相似度**（例如颜色、纹理、特征相似度）。

**Min-Cut 的目标**：找到一条割线，切断的边的权重之和最小。
$$\text{cut}(A, B) = \sum_{u \in A, v \in B} w(u, v)$$

**致命缺陷**：Min-Cut 倾向于切分出**孤立的微小区域**。
例如：把一个孤立的噪点和整张图切开，割边的权重和通常非常小（因为只切断了几条边），这在数学上是“最优”的，但在图像分割中是完全错误的（我们想要的是把“物体”和“背景”分开，而不是切一个点出来）。

### (2) Ncut 的核心改进：归一化 (Normalization)
Jianbo Shi 和 Jitendra Malik 在 2000 年的论文中提出了 Ncut。它的核心思想是：**不仅要求割断的边权重小（差异大），还要求切分出的两个子图内部连接紧密（体积大）。**

Ncut 的目标函数引入了分母来做归一化：
$$\text{Ncut}(A, B) = \frac{\text{cut}(A, B)}{\text{assoc}(A, V)} + \frac{\text{cut}(A, B)}{\text{assoc}(B, V)}$$

* $assoc(A, V)$：集合 A 中所有节点与图中所有节点的连接总和（可以理解为 A 的“体积”或“总度数”）。

**直观理解**：如果 A 只是一个孤立小点，$assoc(A, V)$ 会非常小，导致分数变大（惩罚项）。只有当 A 和 B 都有相当的规模时，Ncut 值才能达到最小。

### (3) 数学求解：广义特征值问题
直接最小化 Ncut 是一个 **NP-Hard** 问题（因为那是离散的组合优化）。
作者使用 **谱松弛 (Spectral Relaxation)** 将其转化为连续空间的问题。

经过数学推导，最小化 Ncut 等价于求解以下**广义特征值问题**：
$$(D - W)y = \lambda D y$$

* $W$：相似度矩阵（$N \times N$）。
* $D$：度矩阵（对角矩阵，对角线元素是 $W$ 每一行的和）。
* $D - W$：即图的 **拉普拉斯矩阵 (Laplacian Matrix)**。
* $y$：特征向量。

**结论**：
1.  最小的特征值是 0，对应的特征向量没用（全是 1）。
2.  **第二小的特征值对应的特征向量 (Second Smallest Eigenvector)** 就是我们要的解！。
3.  这个特征向量是一个连续的实数向量，利用它（比如以 0 或平均值为阈值）就可以把图像切成两半。这也是为什么 TokenCut 代码里要把特征向量可视化的原因——它本质上就是分割的“指示图”。

---

## 2. Fast Normalized Cut (FastNCut): 引入约束与加速

### (1) 为什么要 Fast Ncut？
Ncut 虽然效果好，但计算代价巨大。
求解 $N \times N$ 矩阵的特征值，时间复杂度通常是 $O(N^3)$ 或 $O(N^2)$（使用 Lanczos 方法）。
在 TokenCut 这种需要处理大量图片或高分辨率特征的任务中，原本的 Ncut 太慢了。

### (2) FastNCut 的核心：线性约束 + 投影幂法
FastNCut (Xu et al., 2009) 的核心思想是：我们通常有一些**先验知识 (Prior Knowledge)**，比如“这几个像素肯定属于背景”或“这几个像素肯定属于同一类”。

它把问题转化为了一个**带线性约束的优化问题**：
$$\min \text{Ncut} \quad \text{subject to} \quad Bg = c$$

* $Bg=c$：这是一组线性约束，用来强制某些节点必须属于同一类，或者某些节点的某种组合必须满足特定值。

### (3) 优化算法：Projected Power Method
FastNCut 不需要解完整的特征值方程，而是使用了一种迭代算法：**投影幂法 (Projected Power Method)**。

正是在 `fastncut.py` 中看到的 `_projected_powermethod` 函数的原理：
1.  它不需要对大矩阵做特征分解。
2.  它通过迭代 $v_{k+1} = P A v_k$ 的方式逼近解。
    * $A$ 是矩阵。
    * $P$ 是投影矩阵（基于约束 $B$ 计算出来的），用来保证每一步迭代的结果都满足线性约束。

**速度优势**：这种迭代方法收敛很快，而且避免了极其耗时的全矩阵特征值分解，比传统方法快 1-2 个数量级。

---

## 3. TokenCut 总结

TokenCut 结合了上面两者：

1.  **构图 (Graph)**：它不使用原始像素，而是使用 **DINO (Vision Transformer)** 提取的 **Patch Tokens** 作为节点。由于 Token 数量远少于像素数量，图变小了，而且语义特征更强。
2.  **相似度 (Edge)**：Token 之间的余弦相似度构成了 $W$ 矩阵。
3.  **FastNCut 的应用**：
    * 为了进一步加速和稳定分割，代码使用了 `FastNcut` 类。
    * **关键点**：正如我们之前讨论的，它引入了 `const` (约束)，比如强制图像左上角的 Token 属于背景。
    * 通过 `_projected_powermethod` 快速求出满足“左上角是背景”这一约束条件下的**第二小特征向量**。
4.  **分割**：根据这个特征向量的数值（正负或大小），直接把前景物体“抠”出来，生成 Bounding Box。


## 4\. 代码实现与算法优化 (Code Implementation & Optimization)

针对原始 FastNCut 在复杂背景（如人物、复杂纹理）下 bounding box 容易偏移或不准的问题，我们进行了以下算法层面的优化。主要修改集中在 `object_discovery.py` 文件中。

### (1) 优化一：改进先验约束 (Boundary Constraints)

**问题分析**：
原始代码采用了“左上角硬约束”，假设图像左上角的前几个 Patch 一定属于背景。

  * **缺陷**：如果物体（如人的头部）延伸到了左上角，或者左上角有复杂纹理，该约束会导致前景被错误地划分为背景，导致分割“切断”物体。

**改进方案**：
将约束从“仅左上角”改为“图像四周边界”。假设图像的最外圈通常是背景，强迫算法将四周的 Patch 视为同一类，从而将中间的物体“挤”出来。

**代码修改 (`object_discovery.py` -\> `fast_ncut_optimized` 函数)**：

```python
# 修改 const 的生成逻辑
# 原代码: const = np.array([[0, 1], [0, 2], [1, 5], [3, 8], [4, 9]])

# 新逻辑: 约束图像四边的 Patch 属于同一类 (背景)
h, w = dims  # 假设 dims = (h_featmap, w_featmap)
boundary_indices = []

# 1. 添加顶边和底边的相邻连接
for i in range(w - 1):
    boundary_indices.append([i, i+1]) # 顶边
    boundary_indices.append([(h-1)*w + i, (h-1)*w + i + 1]) # 底边

# 2. 添加左边和右边的相邻连接
for i in range(h - 1):
    boundary_indices.append([i*w, (i+1)*w]) # 左边
    boundary_indices.append([i*w + (w-1), (i+1)*w + (w-1)]) # 右边

const = np.array(boundary_indices)
```

-----

### (2) 优化二：引入 K-Means 自适应阈值 (Adaptive Thresholding)

**问题分析**：
原始代码使用特征向量的 **平均值 (Mean)** 作为二值化的阈值。

  * **缺陷**：平均值对数据分布非常敏感。如果背景区域很大，平均值会偏向背景，导致分割出的前景包含大量噪声。

**改进方案**：
使用 **K-Means 聚类** 替代平均值阈值，自动将特征向量聚类为“前景”和“背景”两类，适应性更强。

**关键修复**：在实现时必须注意变量作用域，需将 `seed` 的计算提前，避免 `UnboundLocalError`。

**代码修改 (`object_discovery.py` -\> `fast_ncut_optimized` 函数)**：

在文件头部添加：

```python
from sklearn.cluster import KMeans
```

在函数内部替换二值化逻辑：

```python
    # ... (前文代码: model.fit 算出 fast_ncut_result) ...
    fast_ncut_result = model.fit(A_numpy)
    second_smallest_vec = fast_ncut_result.reshape(-1)

    # ------------------ 优化代码开始 ------------------
    
    # 1. 【关键修复】提前计算 seed，避免变量未定义报错
    seed = np.argmin(np.abs(second_smallest_vec))

    # 2. 使用 K-Means 聚成 2 类 (前景/背景)
    # 重塑向量以适应 sklearn 输入格式
    vec_data = second_smallest_vec.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(vec_data)
    labels = kmeans.labels_
    
    # 3. 确定哪一类是前景
    # 假设 seed 点（特征值绝对值最大的点）属于前景
    seed_label = labels[seed]
    
    # 生成二值化掩码 (Bi-partition)
    bipartition = (labels == seed_label)
    
    # 格式对齐：重塑为图像尺寸并转为 float
    bipartition = bipartition.reshape(dims).astype(float)
    
    # 特征向量本身不需要再取反或处理符号，直接使用原始向量即可
    eigenvec = second_smallest_vec 

    # ------------------ 优化代码结束 ------------------

    # 调用 detect_box (注意：原代码中后续重复计算 seed 的行需要删除)
    pred, _, objects, cc = detect_box(
        bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]
    )
    # ... (后续代码保持不变)
```

-----

### (3) 优化三：边缘细化 (Post-Processing)

**思路**：
TokenCut 生成的 Mask 是基于 Patch 的（低分辨率），边缘呈锯齿状。为了提升 IoU，我们在生成 Mask 后引入 **Bilateral Solver (双边求解器)**。

  * 利用原始 RGB 图像的色彩信息作为参考，平滑 Ncut 生成的粗糙 Mask，使边缘更贴合物体轮廓。
  * 对应代码库中的 `unsupervised_saliency_detection/bilateral_solver.py`。