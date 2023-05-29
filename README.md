# VoxelFLEX

> NeRF를 활용한 변형체 볼륨 가시화



## Description

VoxelFLEX는 4가지의 ngp(neural graphics primitives) 중 하나인 NeRF(Neural radiance fields)를 기반으로 학습된 3D 모델에 볼륨 렌더링 엔진을 추가하여, **실시간으로 모델을 직접 변형**시킬 수 있도록 한다.

지금까지 영상의학과 관련하여 시술 상담을 진행할 때, 시술 이전과 이후를 비교한 예상 결과를 보여주었는데, 이는 정적인 2D영상으로 결과를 제공하여 한정된 정보만 표현이 가능한 어려움을 지니고 있었다. VoxelFLEX는 이러한 한계를 극복하기 위해 2D영상을 NeRF를 통해 3D영상으로 변환하여, 이전보다 입체적이고 직관적인 시각 정보를 제공할 수 있도록 한다. 또한, 실시간으로 영상 데이터를 변형시킬 수 있어 시술 이전과 이후의 차이를 보다 구체적으로 전달하여 원활한 의사소통을 할 수 있도록 도와준다.

영상의학에서 사용되는 시술 상담은 단순한 예시일 뿐, VoxelFLEX는 동영상을 통해 대상을 촬영할 수 있는 모든 곳에 적용시킬 수 있는 확장성을 지니고 있다. 다음 그림과 같이 사람을 대상으로 하지 않은 인형을 촬영하여 표정을 변형시킬 수도 있다.

<table>
    <thead>
        <tr>
            <th style="text-align: center">Before</th>
        	<th style="text-align: center">After</th>
        </tr>
    </thead>
    <tbody>
    	<tr>
        	<th style="text-align: center"><img src="https://github.com/HSUProject/BuFF/tree/main/docs/assets_readme/Before01.jpg" alt="Before" style="zoom:80%;" /></th>
            <th style="text-align: center"><img src="https://github.com/HSUProject/BuFF/tree/main/docs/assets_readme/After01.jpg" alt="After" style="zoom:80%;" /></th>
        </tr>
    </tbody>
</table>




## How To Build This Project

### Requirements

- **NVIDIA GPU**
- Visual Studio 2019 or 2022
- CUDA 11.5 이상
- CMake v3.21 이상
- Python 3.7 이상



### Compilation

Windows PowerShell 또는 Command Prompt를 실행하여 아래 명령어 또는 Git을 통해 프로젝트를 복제

```shell
$ git clone --recursive https://github.com/HSUProject/BuFF
$ cd BuFF
```



복제한 프로젝트 폴더로 이동 후, 아래 명령어를 통해 프로젝트를 빌드

```shell
BuFF$ cmake . -B build
BuFF$ cmake --build build --config RelWithDebInfo -j
```



> **Troubleshooting**
>
> 컴파일 과정에서 오류가 발생하는 경우 <a style="text-decoration: none" href="https://github.com/NVlabs/instant-ngp#troubleshooting-compile-errors">여기</a>에 흔히 발생하는 오류가 정리되어 있으며, 이외의 오류가 발생하는 경우 CUDA 또는 CMake를 재설치하는 것을 권장한다.



마지막으로 입력 영상을 3D로 변환하기 위해 다음 명령어를 실행

```shell
BuFF$ pip install -r requirements.txt
```



## How To Run This Project

**Step 1. 촬영**

변형하고 싶은 대상을 전체적으로 약 15초 ~ 25초 동안 촬영한다. 이는 최소한의 시간으로 변형할 수 있는 환경을 구성하기 위해 제시된 시간이며, 대상을 다양한 각도에서 오래 촬영할수록 더 높은 품질의 학습 결과가 나타난다.

촬영한 동영상은 **바탕화면**에 저장하도록 한다.



**Step 2. 실행 및 변형**

VoxelFLEX.bat 배치 파일을 실행함으로써 기존에 분리되어 있던 입력 과정을 통합하여 3D 모델링 구축부터 머신 러닝까지 빠르고 간편하게 사용할 수 있도록 변경하였다.

프로그램이 실행되면서 NeRF가 자동으로 주어진 영상을 학습하기 시작한다. 대상이 사람인 경우에는 최소 1500번 이상의 `step`이 진행된 다음 학습을 종료시키는 것을 권장한다.

훈련이 끝나면 상단 메뉴 바에서 Rendering 창에 `Target FPS` 값을 조정 (*낮은 값을 가질수록 높은 화질을 선보이나 성능을 다소 떨어질 수 있다*.) 하여 모델의 선명도를 조정할 수 있다. 

- RTX 3060 기준으로 `Target FPS`의 값을 12정도 주는 것이 화질과 성능 사이의 적당한 균형을 이룰 수 있다.

다음 Edit에서 Crop Box 메뉴를 클릭하면 학습된 모델을 잘라내어 원하는 부위만 보일 수 있도록 조정할 수 있는 메뉴가 나타난다.

- X, Y, Z 축에 대하여 각각 설정할 수 있으며, 회전도 시킬 수 있으므로 더 상세한 편집이 가능하다.

여기까지 설정이 되었다면 이제는 본격적으로 변형을 시킬 차례이다. Edit창에서 `Range`와 `Force`를 통해 입력을 통해 가해지는 변형의 범위와 강도를 조절할 수 있다.

![](https://github.com/HSUProject/BuFF/tree/main/docs/assets_readme/voxelscreen.jpg)



## License

This work is made available under the Nvidia Source Code License-NC. Click <a style="text-decoration: none " href="https://github.com/HSUProject/BuFF/blob/main/LICENSE.txt">here</a> to view a copy of this license.
