
# ディープラーニングを用いたセグメンテーション手法の一つであるRefineNetのChainer実装です。

オリジナルの論文にはいくつかの実装例が書かれていますがこのうち、4-cascaded 1-scale RefineNetを実装しています。https://arxiv.org/pdf/1611.06612.pdf

RefineNetを使ってend-to-endのセグメンテーションを行う際にはエンコーダと組み合わせることになりますが、今回はエンコーダ側にはResNet101を使っていて、Chainer実装は以下を使わせていただいています。  
<https://github.com/yasunorikudo/chainer-ResNet/tree/master/v2>

必要となるimagenetで学習済みのResNetのweightファイルもここで公開されています。ちなみにエンコーダは簡単にResNet50やResNet152に変更することができます。

解析結果をアルファブレンドして表示する部分などで  
<https://github.com/k3nt0w/chainer_fcn>

モデルの一部を取り込むコードとして以下で紹介されている方法を使っています。  
<https://qiita.com/tabe2314/items/6c0c1b769e12ab1e2614>

学習には、オリジナルの画像ファイルと対応するアノテーション済み画像が必要で、それぞれJPEG(xxx.jpg)、PNG(xxx.png)のフォーマットを想定しています。対応するペアのファイル名(xxxの部分)は一致している必要があります。xxxの部分をリストしたテキストファイルがあわせて必要です。

アノテーション済み画像はindex formatのPNG(Pascal VOC2012など)やGrayScaleで学習データが提供されているもの(ADE20Kなど)についてはコードそのまま読めるはずです。data augmentationとしてはhorizontal flipとrandom scale/cropのコードをいれてあります。

引数の定義は以下

-g GPU番号
-tt データリストテキストファイル
-tr オリジナル画像ディレクトリ
-ta アノテーション画像ディレクトリ
-e エポック数
-b バッチサイズ
--lr 学習率

OptimizerにAdamを使っているので学習率はアルファの値です。10エポック、50エポック、以降100エポックごとに学習率を1/10にしています。10エポックごとにweightファイルを吐き出すようにしてあります。

python train.py -g 1 -tt  data/feel.txt -tr ./data/feel_images/ -ta ./data/feel_classes/ -e 500 -b 16 --lr=0.0001
python predict.py -g 1 -i ./data/voc2012_images/2011_003256.jpg -w weight/chainer_refinenet_140.weight

定量評価などはまだ行なっていないのですが、以下はVOC2012で140エポック学習した重みを使った推定例です。

論文の解釈や実装が間違っている部分にお気づきの方はご指摘いただければありがたく思います。