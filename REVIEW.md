# Review : FishNet (A Versatile Backbone for Image, Region, and Pixel Level Prediction)

### Reference : https://arxiv.org/abs/1901.03495
<br/>

---  

## Abstract
<br/>

서로 다른 레벨(Image, Region, Pixel) 의 물체를 예측하는 CNN 구조를 디자인하는 기본적인 원리는 나눠지고 있다.

일반적으로 Detection과 Segmentation에서 쓰이는 backbone 구조 : 분류 task를 위해 디자인된 네트워크 구조

> 그러나, Pixel/Region 레벨(High resolution의 deep feature가 필요한) 네트워크의 장점을 통합하기 위해 디자인된 backbone은 거의 없다. <br/>
> 이러한 목표아래, 해당 논문에서는 물고기같은 네트워크를 디자인 : FishNet

FishNet에서 모든 해상도에 대한 정보는 보존되며, 마지막 task를 위해 정제된다.

추가적으로, 현재의 다른 구조들이 여전히 gradient information을 (from deep to shallow layers) 직접적으로 전파할 수 없다는 것을 발견

> FishNet을 통해 gradient information 문제를 (자연스럽게) 해결 <br/>
> 특히 ImageNet-1k에서, FishNet의 Accuracy : ResNet/DensNet(기존의 gradient preserving network)보다 적은 파라미터로 성능을 능가

<br/>

FishNet은 COCO detection 2018에서 수상한 모듈중 하나에 적용됨
<br/>

---
## 1. Intro
<br/>

CNN : CV분야의 feature representation에 효과적

CNN이 점점 깊어지면서, 최근 연구들은 이전층의 feature를 정제하거나 재사용하려 노력중 : Identitiy mappings or Concatenation

<br/>

Image/Region/Pixel (편의를 위해 I/R/P라고 명명하겠습니다) 레벨의 task를 위해 디자인된 CNN은 네트워크 구조가 나뉘기 시작

* I level

> 분류를 위한 네트워크는 연속적인 다운샘플링을 이용 : 저해상도의 deep feature를 얻기 위함 <br/>
> 그러나 이러한 저해상도의 feature는 R/P level task에 부적합

<br/>

그렇다고 R/P task를 위해 고해상도의 shallow feature를 직접적으로 이용하는 것 역시 좋지 않다.

* P level

> level을 위한 네트워크 구조들은 U-Net or Hourglass같은 네트워크를 이용 : 고해상도의 deep feature를 얻기 위함

<br/>

* R level

> 최근의 Object detection과 같은 R level 연구들은 업샘플링 메커니즘을 이용 : 작은 물체들도 '상대적'으로 고해상도의 feature에 의해 설명 가능

<br/>

R/P level을 위해 고해상도 feature를 이용하는 것의 성공을 통해, 이 논문에서는 물고기와 같은 network를 제안

> 고해상도의 feature가 high-level semantic information을 얻게 하기 위해 <br/>
> 이러한 방식을 통해 이미지 분류로부터 프리트레인된 feature를 R/P task에 더 쉽게 이용 가능

<br/>

---
![image](https://user-images.githubusercontent.com/92928304/169489913-86990ce7-2d0f-4a18-9319-d270c72d99f5.png)
<br/>

* 위와 같은 목표를 위해 세가지의 이점을 가진 메커니즘을 디자인
1. I/R/P task 네트워크들의 이점을 융합한 첫 backbone 네트워크
 <br/>
 
2. Deep layer의 gradient를 shallow layer로 '직접' 전파 : Direct BP (back propagation)

> Direct BP를 가능하게 한 최근의 연구 : Identitiy mapping with residual block and concatenation (ResNet/DensNet) <br/>
> 그러나, 이러한 네트워크들도 사실 여전히 direct BP는 불가능 <br/>
> 불가능한 이유는 서로 다른 해상도의 feature 사이에 있는 conv layer에 의해 발생

<br/>

그림 1.에서 볼 수 있듯이, ResNet은 stride를 가진 conv layer를 skip connection에 이용 : 채널을 통일하기 위함

> 저자는 이것이 진정한 Identitiy mapping이 안되는 이유라고 주장합니다. <br/>
> 그럼에도 이러한 방식을 쓰지 않는 Convolution의 경우, output부터 shallow layer까지의 gradient를 감소시키는 문제가 발생
 
 <br/>
 
FishNet은 이러한 문제를 깊이가 상당히 차이나는 feature들의 concatenation을 통해 더 잘 해결할 수 있으며,

direct BP를 보장하기 위해 네트워크의 요소들을 아주 조심스럽게 디자인 : feature의 의미론적 의미가 전체 네트워크에서 유지

<br/>

3. 깊이 차가 많이 나는 feature가 보존되며 서로를 정제하기 위해 이용됨

서로 다른 깊이에서 나온 feature : 이미지 축약의 단계 또한 상이하며, 모든 단계의 정보는 feature diversity를 위해 유지되어야함

> 다른 깊이의 feature들은 상호보완성을 가지며, 이것이 서로를 정제하는데 이용될 수 있음 <br/>
> 따라서, Feature Preserving and Refining machanism이 연구의 목적
---
<br/>

직관과 반대되는(?) FishNet의 효과는, 기존의 conv network (parameter와 accuracy간의 trade-off를 가지는 classification network)보다 더 성능이 우수하다는 점이다.

이유는 아래와 같다.

> 1. 보존/정제된 feature들이 서로 상호보완적이며, width/depth를 늘리는 것보다 효과적이다.
> 2. Direct BP가 가능 : 실험 결과는 FishNet150(ResNet50과 같은 파라미터를 가짐)이 ResNet101, DensNEt161(k=48)의 acc.를 능가할 수 있음을 보여줌 (On ImageNet-1k)

<br/>

R/P level task (객체 감지, instance segmentation과 같은)에 대해서,

fishnet은 Mask R-CNN의 backbone으로 적용되어 resnet baseline보다 AP 2.8% (2.3%)를 향상 (on MSCOCO)


<br/>

### 1.1 Related works

* 이미지 분류 CNN 아키텍쳐

기존 CNN 아키텍쳐에 대한 설명 (중략)

Vanishing gradient를 해결하기 위한 Skip connection 관련 연구 : 이미지 분류를 위한 네트워크이며

> 고해상도의 feature를 small receptive field를 가진 shallow layers로부터 추출 <br/>
> Deep layer로밖에 얻을 수 없는 High level semantic meaning이 부족<br/>
> 연구의 목적은 고해상도의 deep feature를 high-level semantic meaning과 함께 추출하고, 분류 Acc.를 동시에 향상시키는 것
<br/>

![image](https://user-images.githubusercontent.com/92928304/169498733-8558b10d-65f5-443c-8643-084154279342.png)
<br/>

---
그림에 대해서 설명을 하자면 네트워크를 Tail, Body, Head의 세 부분으로 분리하여 설명합니다.

Tail은 기존의 분류 task와 같이 Deep low resolution feature를 추출

Body는 tail과의 feature preserving/refining을 통해 high resolution feautures with high level semantic information를 추출

> 말이 좀 복잡해보이지만, FPN처럼 낮은 단계의 feature를 body에서 concat하면서 학습하여 high level semantic이 유지된다 라는 의미인 것 같습니다.

Head는 앞선 feature들을 역시 보존/정제하고 task를 수행합니다.

---

* Design in combining features from different layers (저자는 양방향 refinement를 위해 노력한 것 같습니다)

다른 해상도/깊이의 feature : Nested sparse network, hyper-column, addition, residual block을 이용하여 결합이 가능

> Hyper column[1] : 서로 다른 layer의 feature를 직접 conatenate, But layer 사이에 상호작용은 X<br/>
![image](https://user-images.githubusercontent.com/92928304/169502901-7ae71ed3-67d9-4ffe-8e83-0a63bfac0a3b.png)

> Addition : 단순 더하는 방법, 역시 deep/shallow를 보존하거나 서로 정제할 수는 없음 (ex. ResNet의 shallow feature는 보존되지 X)<br/>
> Concatenation : DensNet<br/>

* Networks with up-sampling mechanism

Classification 이외의 Computer vision에서(Detection, Segmentation) upsample은 필수적이며, 서로 다른 깊이의 layer 간의 커뮤니케이션을 포함

> U-Net, FPN, stacked hourglass etc<br/>
> 이러한 알고리즘들이 분류 task에 효과적임은 입증된적 없다.

MSDNet은 large resolution(여러 depth를 의미하는 듯)의 featuremap을 유지하기 위해 노력 : FishNet과 유사한 구조

> 그러나, MSDNet 역시 서로 다른 해상도 간에 conv를 이용 : Representation을 보존할 수 없음!!<br/>
> 또한, Feature가 다양한 resolution과 semantic meaning을 가질 수 있게 만드는 upsample pathway가 부재<br/>
(Budget prediction을 위한 multi-scale mechanism을 소개하는 것이 목표였다고 주장)

FishNet의 차이점은 다음과 같다.

> MSDNet 역시 분류 task에 있어 정확도 향상을 보여주진 못함 : FishNet은 처음으로 U-Net structure가 분류에 효과적임을 보여줌<br/>
> Shallow/Deep layer 모두 보존하고 정제되어 final task에 적용 : 기존의 Upsample 네트워크와 다르다.
<br/>

* Message passing among features/outputs

Segmentation[36], Pose estimation[3], Object detection[35] = Feature 사이의 Message passing을 이용하는 접근 방법들

> 이 디자인들은 모두 Backbone 네트워크에 기반하며, fishnet은 이를 보완할 수 있음 (뒤에서 자세히 나옵니다)
---
<br/>

## 2. Identitiy Mappings in Deep Residual Networks and Isolated Convolution

레스넷을 구성하는 기본 block은 residual block

(1) Residual blocks with identitiy mapping
![image](https://user-images.githubusercontent.com/92928304/170251745-cacc1b54-2fa4-4897-bca4-2cf5cf9778b3.png)

> xl은 layer l에서 residual block에 들어가는 input feature<br/>
> F(xl, Wl) : residual function
<br/>

본 논문에서 같은 해상도를 갖는 residual block의 stack을 Stage(s)라고 명시합니다.

(2) Stacked res block by stage
![image](https://user-images.githubusercontent.com/92928304/170252259-37193ffa-9893-4d4e-8216-08adc1a6727f.png)
(좌측식을 편미분하면 자연스럽게 우측식이 나오고, 1이 생김으로써 마지막 layer의 gradient가 사라지지 않고 bp되어 자연스럽게 residual을 학습할 수 있게됩니다)
<br/>

여기서 저자는 서로 다른 스테이지에 있는 (다른 해상도를 갖는) feature에 대해 고민합니다.

그 이유는 레스넷에서, 다른 해상도의 feature는 다른 채널수를 가지기 때문.

> 따라서, downsample을 하기 전, 채널을 조절하기 위한 transition function h(ㆍ)이 필요

(3) Transition function : 이전 stage X(Ls,s)에서 다음 stage X'(0,s+1)로 downsample하는 과정 (같은 stage에서 첫 layer는 0, 마지막은 Ls로 표기)
![image](https://user-images.githubusercontent.com/92928304/170253082-2687ffe1-08b1-46d6-826f-91c268a81631.png)

<br/>

* Gradient propagation problem from Isolated convolution (I-conv)

I-conv : identitiy mapping or concat이 없는 convolution (resolution이 바뀔 때 단순히 conv1x1 with stride2로 바꿔주는 layer를 의미하는 것 같습니다.)

(이전의 연구에 따르면, gradient는 deep에서 shallow로 direct propagation되는 것이 바람직하다고 합니다.)

I-conv는 gradient가 직접적으로 전달되지 못하게 방해 : Resnet에서 서로 다른 해상도의 feature, Densnet에서 인접한 denseblock

Stage 내의 모든 feature를 이용하는 Invertible down-sampling[2]은 I-conv의 문제를 피할 수 있지만, stage ID가 늘수록 파라미터수도 기하급수적으로 상승


<br/>

## 3. The FishNet

이제 본격적으로 모델 구조에 대해서 설명합니다.

![image](https://user-images.githubusercontent.com/92928304/170255515-ae9e8cdc-dbc3-4c53-9750-8b13cb23131d.png)
---
Tail : ResNet과 같은 기존의 CNN 네트워크이며, 깊어짐에 따라 resolution이 감소

Body : 여러 Upsampleing & Refining block들을 가짐, tail과 body로부터 feature를 refine

Head : 여러 Downsampling & Refining block, final task에 이용

---

Stage : 같은 해상도의 feature를 공급받는 conv block 뭉치

Fishnet의 각 파트는 "output의 resolution에 따라" 여러 스테이지로 나뉨 : 해상도가 작아질수록, ID는 커짐

본 연구에서는 56/1(해상도/stage), 28/2, 14/3, 7/4 로 모델을 구성

![image](https://user-images.githubusercontent.com/92928304/170258138-fbb1fbe8-c02e-4ccb-b364-86d8efc76150.png)

그림 3은 두 스테이지에서 feature 사이의 상호작용을 보여준다. (a)의 fish tail은 residual network로 간주한다.

꼬리의 feature는 여러 residual block을 겪고 body로 전달됨

바디의 경우 concat을 통해 tail과 이전 스테이지의 feature를 보존

> Concated features는 upsample되며 (b)의 과정에 의해 refined (자세한건 섹션 3.1.)<br/>
> 정제된 피쳐는 body의 다음 스테이지와, head에 이용됨

헤드는 바디의 모든 feature와 이전 스테이지의 feature를 보존/정제 : 정제된 feature가 다음 스테이지로 이동

<br/>

그림 3(c)는 헤드에서 진행되는 message passing을 나타냄 (섹션 3.1.)

Horizontal connections는 transferring block를 의미하며, 본 연구에서는 residual block을 적용 (코드 구현 시 transfer에 residual block을 적용해야함)

<br/>

## 3.1 Feature refinemnet

fishnet은 Up-sampling & Refinement block (UR-block)과 Down-sampling & Refinement block (DR-block)의 block을 이용

* UR block
![image](https://user-images.githubusercontent.com/92928304/170262288-0776e938-7b41-4c3a-ac3f-168a02189591.png)

위의 두 변수는 각각 tail/body에서 "first layer의 output feature"를 의미 = 해당 스테이지(resolution)가 된 첫 x를 의미 = 해상도가 변한 x

(4~8)
![image](https://user-images.githubusercontent.com/92928304/170267919-cb755ebd-c8ee-4485-93bf-7b9941f81b00.png)
![image](https://user-images.githubusercontent.com/92928304/170267950-e1561cfc-1b0b-4f49-837d-90c94654f67e.png)
![image](https://user-images.githubusercontent.com/92928304/170268190-ebabe4c6-1ac6-41d5-b1ed-c89228f739af.png)

8 : channel-wise reduction, 채널을 k개 단위로 잘라서 더해준다.
7 : concat
6 : refine
5 : upsample

M : bottleneck residual block

* DR block

(9)
![image](https://user-images.githubusercontent.com/92928304/170268448-52d0c183-7220-462f-930b-c0d2579e28e1.png)

2x2 maxpool을 이용하며, downsample시 channel reduction을 수행하지 않습니다.

> 이전 스테이지의 gradient가 직접적으로 전달<br/>
> 이를 통해 진정한 residual block이라고 주장합니다.<br/>
> (제 생각에는 body에도 reduction function을 적용해야 direct propagation이 될 것 같은데, 파라미터 수의 문제인지 확인해봐야 알 것 같습니다.)










# Reference
[1] Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 447-456

[2] J.-H. Jacobsen, A. Smeulders, and E. Oyallon. i-revnet: Deep invertible networks. arXiv preprint arXiv:1802.07088, 2018.
