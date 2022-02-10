# cuBLASamples
#### NVIDIA cuBLAS examples

The cuBLAS Library provides a GPU-accelerated implementation of the basic linear algebra subroutines (BLAS).

![](https://tse1-mm.cn.bing.net/th/id/R-C.f5a16b8b901a616c78fdecfdfd77673f?rik=Ic18ZGcEhTtSlA&riu=http%3a%2f%2fwww.jihosoft.com%2fitunes%2fgpu-acceleration-technique.png&ehk=o2qU8y7JncVtrVgAI%2bp%2f5DF0epfP%2bmJvHru5L5Aezh8%3d&risl=&pid=ImgRaw&r=0)

In this project, You can create an matrix on GPU just call:

```c++
cuMatrix32f A(rows, cols);
```

or create an vector on GPU:

```
cVector<float> A(len);
```

import data from host

```c++
A.copyFromHost(cpu_data);
```

export data to host

```c++
A.copyToHost(cpu_data);
```

and do matrix multiplication C = A * B

```c++
A.matmul(B,C);
```

##### GPU MEMORY MANAGED INSIDE AND FREE AUTOMATICALLY

```c++
template<typename T>
class cuVector
{
public:
    cuVector(int len) { /*MALLOC DATA ON GPU AUTOMATICALLY*/ }
    ~cuVector() { /*AND FREE IT AUTOMATICALLY*/ }
}
```

##### PERFORMANCE

In GTX1060, Repeat **C=A*B** for 10000 times cost 58ms

![](https://images2017.cnblogs.com/blog/1252943/201710/1252943-20171031201649513-1955183952.gif)
