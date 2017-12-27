# chainer_refinenet

ディープラーニングを用いたセグメンテーション手法の一つであるRefineNetのChainer実装です。

[オリジナルの論文](https://arxiv.org/pdf/1611.06612.pdf)にはいくつかの実装例が書かれていますがこのうち、**4-cascaded 1-scale RefineNet**を実装しています。

RefineNetを使ってend-to-endのセグメンテーションを行う際にはエンコーダと組み合わせることになりますが、今回はエンコーダ側にはResNet101を使っていて、Chainer実装は[こちら](https://github.com/yasunorikudo/chainer-ResNet/tree/master/v2)を使わせていただいています。 

必要となるimagenetで学習済みのResNetのweightファイルも[こちら](https://github.com/yasunorikudo/chainer-ResNet/tree/master/v2)からダウンロードできます。ちなみにエンコーダは簡単にResNet50やResNet152に変更することができますのでご興味がある方はお試しください。

いろいろな方のコードを参考にさせていただいていますが、特にChainerのモデルの一部を取り込むコードとして[こちら](https://qiita.com/tabe2314/items/6c0c1b769e12ab1e2614)で紹介されている方法を使っています。また解析結果をアルファブレンドして表示する部分などで[こちら](https://github.com/k3nt0w/chainer_fcn)のコードの一部を使わせていただいています。

## 使い方
Python3.x、Chainer3.xで動作確認しています。その他

pip install pillow  
pip install opencv-python  
pip install cupy  

で必要なモジュールが入ると思います。

学習には、オリジナルの画像ファイルと対応するアノテーション済み画像が必要で、それぞれJPEG(xxx.jpg)、PNG(xxx.png)のフォーマットを想定しています。対応するペアのファイル名(xxxの部分)は一致している必要があります。また、xxxの部分をリストしたテキストファイルがあわせて必要です。アノテーション済み画像はindex formatのPNG(Pascal VOC2012など)やGrayScaleで学習データが提供されているもの(ADE20Kなど)についてはコードそのまま読めるはずです。data augmentationとしてはhorizontal flipとrandom scale/cropのコードをいれてあります。

オリジナルファイルとアノテーションファイルが以下のようにあるとすれば  
images/aaa.jpg  
images/bbb.jpg  
images/ccc.jpg  
annotations/aaa.png  
annotations/bbb.png  
annotations/ccc.png  

リストファイル、例えばtrain.txtの中身を以下のようにします。  
aaa  
bbb  
ccc  

学習を行うコマンドは以下のようになります。  
`$ python train.py -g 1 -tt  train.txt -tr ./images/ -ta ./annotations/ -e 500 -b 16 -n 21 -l 0.0001`

引数の定義は以下  
-g GPU番号  
-tt データリストテキストファイル  
-tr オリジナル画像ディレクトリ  
-ta アノテーション画像ディレクトリ  
-e エポック数  
-b バッチサイズ
-n クラス数  
-l 学習率  

OptimizerにAdamを使っているので学習率はalphaの値です。10エポック、50エポック、以降100エポックごとに学習率を1/10にしています。10エポックごとにweightファイルを吐き出すようにしてあります。

## 実行例
定量評価などはまだ行なっていないのですが、以下はVOC2012で140エポック学習した重みを使った推定例です。

`$ python predict.py -i ./data/voc2012_images/2011_003242.jpg -g 1 -n 21 -w weight/chainer_refinenet_140.weight`

引数の定義は以下  
-g GPU番号  
-w 学習済みのウェイトファイル  
-n クラス数  
-i 推定対象の画像  

prediction (テストデータ)  
![推定結果](https://raw.githubusercontent.com/ponta256/images/master/2011_003242_pred.jpg)

original (テストデータ)  
![オリジナル画像](https://raw.githubusercontent.com/ponta256/images/master/2011_003242_origin.jpg)

`$ python predict.py -i ./data/voc2012_images/2011_003256.jpg -w weight/chainer_refinenet_140.weight`

prediction (学習データ)  
![推定結果](https://raw.githubusercontent.com/ponta256/images/master/2011_003256_pred.jpg)

Ground Truth (学習データ)  
![Ground Truth](https://raw.githubusercontent.com/ponta256/images/master/2011_003256_ground_truth.png)

original (学習データ)  
![オリジナル画像](https://raw.githubusercontent.com/ponta256/images/master/2011_003256_origin.jpg)

論文の解釈や実装が間違っている部分にお気づきの方はご指摘いただければありがたく思います。
