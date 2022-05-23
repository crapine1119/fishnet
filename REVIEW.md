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











# Reference
[1] Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 447-456