# NEU_ML_Course
从UCI网站中选取不均衡数据集，进行不均衡数据的实验分析，可从以下几个研究点（不需要全做，也不限于以下几个点）进行探索，并完成一个报告。

1. 不均衡数据对传统分类器的影响

2. 不同采样算法的效果对比及分析

3. 针对SMOTE算法的问题，可有针对性地提出新的解决f方案

4.  针对多标签不均衡数据的问题，如何扩展SMOTE算法。

会根据报告的选题角度，实验的数量，分析的深度，创新性和报告写作的规范性，进行综合打分。

1 不均衡数据集的选择及获取

1.1 不均衡数据集的来源

访问UCI机器学习库网站(https://archive.ics.uci.edu/ml/index.php)。点击“数据集”，然后点击“分类”，以查看可用的不均衡数据集。

之后可以选择一个不均衡数据集，然后点击“数据集”以下载。

任务种类包括分类(classification)、回归(regression)、集群(clustering)。
鉴于作业要求探究不均衡数据对传统分类器的影响，因此选择分类任务的数据集。
可供选择的UCI网站上的466个数据集中选择，其中不均衡数据集包括：
Credit Card Fraud Detection Data Set：这是一个信用卡欺诈检测数据集，其中正例占比非常小，只有0.172%，非常不均衡。
Bank Marketing Data Set：这是一个银行营销数据集，其中正例（即客户订购定期存款产品）占比仅为11.27%，非常不均衡。
Statlog (German Credit Data) Data Set：这是一个德国信用数据集，其中正例（即客户有良好的信用）的比例为30.87%，虽然不是非常不均衡，但仍然相对较小。
Banknote authentication Data Set：数据是从真钞和伪造钞票样标本中获取的图像中提取的。该数据集包含1372个样本，每个样本有4个属性，分别是：Variance of Wavelet Transformed image、Skewness of Wavelet Transformed image、Kurtosis of Wavelet Transformed image和Entropy of image。最后一列是目标变量，1表示真实的银行票据，0表示伪造的银行票据。其中真实的银行票据占比为0.62，而伪造的银行票据占比为0.38。

1.2 不均衡数据集的选择

由于我在植物分类中处理了图像，因此选择同样需要图像处理的Banknote authentication Data Set。后来下载完发现不需要处理图像，数据在一个txt中。

2基于不均衡数据集的实验分析

接下来使用这些数据集来进行不均衡数据的实验分析，比如使用不同的数据处理方法（如数据重采样、数据加权、特征选择等）来处理不均衡数据，并对不同处理方法的效果进行比较分析。

2.1 imblearn算法介绍

2.1.1 过采样(Over)

RandomOverSampler

原理：从样本少的类别中随机抽样，再将抽样得来的样本添加到数据集中。
缺点：重复采样往往会导致严重的过拟合。
主流过采样方法是通过某种方式人工合成一些少数类样本，从而达到类别平衡的目的，而这其中的鼻祖就是SMOTE。

SMOTE

原理：在少数类样本之间进行插值来产生额外的样本。对于少数类样本a, 随机选择一个最近邻的样本b, 从a与b的连线上随机选取一个点c作为新的少数类样本;
具体地，对于一个少数类样本xi使用K近邻法(k值需要提前指定)，求出离xi距离最近的k个少数类样本，其中距离定义为样本之间n维特征空间的欧氏距离。然后从k个近邻点中随机选取一个，生成新样本。
SMOTE会随机选取少数类样本用以合成新样本，而不考虑周边样本的情况。

BorderlineSMOTE

这个算法会先将所有的少数类样本分成三类，noise： 所有的k近邻个样本都属于多数类；danger： 超过一半的k近邻样本属于多数类；safe： 超过一半的k近邻样本属于少数类。

2.1.2 欠采样(Under)

欠采样：欠采样是一种采样算法，它会从多数类中移除一些样本以减少样本数量，以使其与少数类的样本数量相匹配。这种方法可以有效地改善数据的不均衡性，但是它可能会丢失一些有用的信息。

RandomUnderSampler
随机选取数据的子集

TomekLinks
样本x与样本y来自于不同的类别, 满足以下条件, 它们之间被称之为TomekLinks:不存在另外一个样本z, 使得d(x,z) < d(x,y) 或者 d(y,z) < d(x,y)成立. 其中d(.)表示两个样本之间的距离, 也就是说两个样本之间互为近邻关系. 这个时候, 样本x或样本y很有可能是噪声数据, 或者两个样本在边界的位置附近。

2.1.3 过采样与欠采样结合（combine）

混合采样：混合采样是一种采样算法，它会同时使用过采样和欠采样的方法来改善数据的不均衡性。这种方法可以有效地改善数据的不均衡性，同时又能够最大程度地保留有用的信息。
SMOTE算法的缺点是生成的少数类样本容易与周围的多数类样本产生重叠难以分类，而数据清洗技术恰好可以处理掉重叠样本，所以可以将二者结合起来形成一个pipeline，先过采样再进行数据清洗。主要的方法是 SMOTE + ENN 和 SMOTE + Tomek ，其中 SMOTE + ENN 通常能清除更多的重叠样本。

2.2 训练结果

使用pandas加载数据，使用sklearn划分数据集，使用StandardScaler()特征归一化。
对于未经处理的数据,我训练了包括KernelRidge、 Ridge、SGD、Lasso、LinearRegression ExtraTrees、GradientBoosting、RandomForest、SVM、XGBoost在内的10个模型。使用Kfold交叉验证。
表 1 不同模型的预测结果
模型	准确率	交叉验证准确率
XGboost	0.995145	0.992708
svm	1.0	1.0
RandomForest	0.878640	0.961611
GradientBoosting	0.699029	0.955441
ExtraTrees	0.839805	0.990899
LinearRegression	0.686893	0.865620
Ridge	0.684466	0.865609
KernelRidge	0.563106	0.043698
Lasso	0.563106	-0.007906
SGDRegressor	0.667475	0.863159

传统分类器在处理不均衡数据集时存在一定的问题。由于不均衡数据集中正例样本数量远小于负例样本数量，因此传统分类器的性能会受到影响。由于正例样本的数量少，传统分类器很难从正例样本中学习到有用的特征，从而导致传统分类器容易出现过拟合现象。此外，由于负例样本的数量多，传统分类器容易倾向于将所有样本分类为负例，从而导致模型的准确率低。 
对于UCI上的banknote数据集而言，由于其不均衡的数据，传统分类器的表现不太好，出现高偏差和高方差的情况。
为了改善这一点，可以通过采用数据增强、重采样或其他方法来改善数据的不均衡性，从而提高传统分类器的性能。

2.3使用过采样处理数据的训练结果

过采样是一种采样算法，它会从少数类中复制一些样本以增加样本数量，以使其与多数类的样本数量相匹配。这种方法可以有效地改善数据的不均衡性，但是它也可能会导致过拟合的问题。
首先使用基于距离的过采样(SMOTE)处理。
SMOTE算法的原理是，对于少数类样本中的每一个样本，都会找到与其最近邻的k个样本，然后从这k个样本中随机选择一个作为合成样本的特征。通过这种方法，可以使得少数类样本的数量达到均衡。SMOTE算法的缺点是，合成的样本可能会带来噪声，因此在使用SMOTE算法之前，需要先对数据进行降维处理，这里使用了PCA算法。
PCA算法的原理是，对于一个n维的数据集，PCA算法会将其降维到k维，其中k<n，这样就可以减少数据集中的噪声。
降维之后，再使用SMOTE算法，使得数据集的样本数目达到均衡。
数据处理后训练结果如表所示。

表 2 不同模型过采样后的预测结果

模型	处理前交叉验证得分	处理后交叉验证得分

XGboost	0.992708	0.978301

svm	1.0	0.936792

RandomForest	0.961611	0.893628

GradientBoosting	0.955441	0.809441

ExtraTrees	0.990899	0.937425

LinearRegression	0.865620	0.223045

Ridge	0.865609	0.222808

KernelRidge	0.043698	-2.445755

Lasso	-0.007906	-1.565639

SGDRegressor	0.863159	0.229067

结果显示，他们的交叉验证得分均有下降，其中XGboost、svm、ExtraTrees小幅下降，LinearRegression、Ridge、KernelRidge、Lasso、SGDRegressor大幅下降。说明数据出现了过拟合。

2.4对比不同的采样算法

选择XGboost、ExtraTrees、KernelRidge、Lasso作为代表的模型，接下来就几个模型对比不同的采样算法。

表 3 四个模型不同采样后的结果

模型	XGboost	ExtraTrees	KernelRidge	Lasso

RandomUnderSampler	0.954651	0.0	0.0	0.0

RandomOverSampler	0.978301	0.943402	-2.852366	-1.565639

TomekLinks	0.974868	0.933808	-0.200101	-0.008716

SMOTETomek	0.975283	0.942651	-2.732367	-1.548371

SMOTEENN	0.984049	0.081127	-0.687793	-0.690526

可以看出，RandomUnderSampler欠采样后ExtraTrees、KernelRidge、Lasso出现明显的过拟合；RandomOverSampler过采样后XGboost、ExtraTrees表现下降，KernelRidge、Lasso大幅下降；TomekLinks欠采样后XGboost、ExtraTrees表现下降，KernelRidge、Lasso表现轻微下降；SMOTETomek组合采样后XGboost、ExtraTrees表现下降，KernelRidge、Lasso出现明显的过拟合，表现大幅下降；SMOTEENN组合采样后XGboost表现下降ExtraTrees、KernelRidge、Lasso表现大幅下降。

XGBoost表现坚挺，这可能是由于其加入树模型复杂度的正则项来抑制树的生长防止过拟合；也说明XGBoost可以很好的应对不平衡训练集。
本数据集采用1表示真实的银行票据，0表示伪造的银行票据。其中真实的银行票据占比为0.62，而伪造的银行票据占比为0.38。过采样在伪造的银行票据中随机抽样，再将抽样得来的样本添加到数据集中；欠采样在真实的银行票据中移除一些样本以减少样本数量，以使其与少数类的样本数量相匹配。
注意到过采样对ExtraTrees影响较小，而欠采样极大的影响了ExtraTrees。同时过采样对KernelRidge、Lasso影响程度大于欠采样。

3 SMOTE算法的改进

3.1 SMOTE算法的问题

SMOTE会随机选取少数类样本用以合成新样本，而不考虑周边样本的情况，这样带来两个问题：如果选取的少数类样本周围也都是少数类样本，则新合成的样本不会提供太多有用信息；如果选取的少数类样本周围都是多数类样本，这类的样本可能是噪音，则新合成的样本会与周围的多数类样本产生大部分重叠，致使分类困难。

3.2 使用kmeans结合SMOTE

聚类、滤波和过采样。在聚类步骤中，使用k-均值聚类将输入空间聚类成k个组。滤波步骤选择用于过采样的聚类，保留少数类样本比例高的聚类。然后分配要生成的合成样本的数量，将更多的样本分配给少数样本稀疏分布的集群。最后，在过采样步骤中，在每个选定的聚类中应用SMOTE，以实现少数和多数实例的目标比率。
