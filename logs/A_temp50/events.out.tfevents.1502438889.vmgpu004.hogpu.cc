       �K"	  @zYc�Abrain.Event:2Z�σ3�     �̘.	��OzYc�A"��
^
dataPlaceholder*
shape: *
dtype0*/
_output_shapes
:���������
W
labelPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������

h
conv2d_1_inputPlaceholder*/
_output_shapes
:���������*
dtype0*
shape: 
v
conv2d_1/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *�x�*
_output_shapes
: *
dtype0
`
conv2d_1/random_uniform/maxConst*
valueB
 *�x=*
_output_shapes
: *
dtype0
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2���*
dtype0*
T0*
seed���)
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0
�
conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
:@*
T0
�
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
:@
�
conv2d_1/kernel
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
�
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
[
conv2d_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_1/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
�
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
s
conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
s
"conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*/
_output_shapes
:���������@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
�
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
v
conv2d_2/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
`
conv2d_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�\1�
`
conv2d_2/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�\1=
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2���*
T0*
seed���)*
dtype0
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0
�
conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
:@@
�
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*&
_output_shapes
:@@*
T0
�
conv2d_2/kernel
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
dtype0*
shared_name 
�
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
[
conv2d_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_2/bias
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
�
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
t
conv2d_2/bias/readIdentityconv2d_2/bias*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
s
conv2d_2/convolution/ShapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
s
"conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*/
_output_shapes
:���������@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0
e
activation_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
a
dropout_1/keras_learning_phasePlaceholder*
_output_shapes
:*
shape: *
dtype0

�
dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
:*
T0

e
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2��
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*/
_output_shapes
:���������@*
T0
�
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:���������@
{
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
T0*
N
c
flatten_1/ShapeShapedropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
g
flatten_1/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
i
flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
i
flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
shrink_axis_mask *
_output_shapes
:*
Index0*
T0*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask 
Y
flatten_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
\
flatten_1/stack/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
_output_shapes
:*
N*

axis *
T0
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
T0*0
_output_shapes
:������������������*
Tshape0
m
dense_1/random_uniform/shapeConst*
valueB" d  �   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�3z�*
_output_shapes
: *
dtype0
_
dense_1/random_uniform/maxConst*
valueB
 *�3z<*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2���*
T0*
seed���)*
dtype0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*!
_output_shapes
:���*
T0
�
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*!
_output_shapes
:���*
T0
�
dense_1/kernel
VariableV2*
shape:���*
shared_name *
dtype0*!
_output_shapes
:���*
	container 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
~
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
z
dense_1/bias
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_output_shapes	
:�*
validate_shape(*
_class
loc:@dense_1/bias*
T0*
use_locking(
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *(
_output_shapes
:����������*
transpose_a( *
T0
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:����������
]
activation_3/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:����������
�
dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
:
e
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��.*
T0*
seed���)*
dtype0
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:����������
t
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
�
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
N*
T0**
_output_shapes
:����������: 
m
dense_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"�   
   
_
dense_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *̈́U�
_
dense_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *̈́U>
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
_output_shapes
:	�
*
seed2���*
dtype0*
T0*
seed���)
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes
:	�
*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	�

�
dense_2/kernel
VariableV2*
shape:	�
*
shared_name *
dtype0*
_output_shapes
:	�
*
	container 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
|
dense_2/kernel/readIdentitydense_2/kernel*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
Z
dense_2/ConstConst*
_output_shapes
:
*
dtype0*
valueB
*    
x
dense_2/bias
VariableV2*
_output_shapes
:
*
	container *
dtype0*
shared_name *
shape:

�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
q
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0
�
dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*'
_output_shapes
:���������
*
data_formatNHWC*
T0
�
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
�
'sequential_1/conv2d_1/convolution/ShapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*
strides
*
data_formatNHWC*/
_output_shapes
:���������@*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
�
'sequential_1/conv2d_2/convolution/ShapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
�
/sequential_1/conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC

sequential_1/activation_2/ReluRelusequential_1/conv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
�
"sequential_1/dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

y
$sequential_1/dropout_1/cond/switch_tIdentity$sequential_1/dropout_1/cond/Switch:1*
_output_shapes
:*
T0

w
$sequential_1/dropout_1/cond/switch_fIdentity"sequential_1/dropout_1/cond/Switch*
T0
*
_output_shapes
:
r
#sequential_1/dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
�
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
&sequential_1/dropout_1/cond/mul/SwitchSwitchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*1
_class'
%#loc:@sequential_1/activation_2/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0
�
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2��H*
dtype0*
T0*
seed���)
�
6sequential_1/dropout_1/cond/dropout/random_uniform/subSub6sequential_1/dropout_1/cond/dropout/random_uniform/max6sequential_1/dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/mulMul@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_1/cond/dropout/random_uniform/sub*/
_output_shapes
:���������@*
T0
�
2sequential_1/dropout_1/cond/dropout/random_uniformAdd6sequential_1/dropout_1/cond/dropout/random_uniform/mul6sequential_1/dropout_1/cond/dropout/random_uniform/min*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/addAdd-sequential_1/dropout_1/cond/dropout/keep_prob2sequential_1/dropout_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:���������@
�
)sequential_1/dropout_1/cond/dropout/FloorFloor'sequential_1/dropout_1/cond/dropout/add*
T0*/
_output_shapes
:���������@
�
'sequential_1/dropout_1/cond/dropout/divRealDivsequential_1/dropout_1/cond/mul-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
'sequential_1/dropout_1/cond/dropout/mulMul'sequential_1/dropout_1/cond/dropout/div)sequential_1/dropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
$sequential_1/dropout_1/cond/Switch_1Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu*
T0
�
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*
N*
T0*1
_output_shapes
:���������@: 
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
_output_shapes
:*
out_type0*
T0
t
*sequential_1/flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,sequential_1/flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
end_mask
f
sequential_1/flatten_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
sequential_1/flatten_1/ProdProd$sequential_1/flatten_1/strided_slicesequential_1/flatten_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
sequential_1/flatten_1/stack/0Const*
_output_shapes
: *
dtype0*
valueB :
���������
�
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:
�
sequential_1/flatten_1/ReshapeReshape!sequential_1/dropout_1/cond/Mergesequential_1/flatten_1/stack*
T0*0
_output_shapes
:������������������*
Tshape0
�
sequential_1/dense_1/MatMulMatMulsequential_1/flatten_1/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
w
sequential_1/activation_3/ReluRelusequential_1/dense_1/BiasAdd*(
_output_shapes
:����������*
T0
�
"sequential_1/dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
y
$sequential_1/dropout_2/cond/switch_tIdentity$sequential_1/dropout_2/cond/Switch:1*
T0
*
_output_shapes
:
w
$sequential_1/dropout_2/cond/switch_fIdentity"sequential_1/dropout_2/cond/Switch*
T0
*
_output_shapes
:
r
#sequential_1/dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
�
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������
�
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
�
-sequential_1/dropout_2/cond/dropout/keep_probConst%^sequential_1/dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
�
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
T0*
_output_shapes
:*
out_type0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/minConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
�
6sequential_1/dropout_2/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
6sequential_1/dropout_2/cond/dropout/random_uniform/mulMul@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_2/cond/dropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
2sequential_1/dropout_2/cond/dropout/random_uniformAdd6sequential_1/dropout_2/cond/dropout/random_uniform/mul6sequential_1/dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/addAdd-sequential_1/dropout_2/cond/dropout/keep_prob2sequential_1/dropout_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:����������
�
)sequential_1/dropout_2/cond/dropout/FloorFloor'sequential_1/dropout_2/cond/dropout/add*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/divRealDivsequential_1/dropout_2/cond/mul-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/mulMul'sequential_1/dropout_2/cond/dropout/div)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu*
T0
�
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul*
T0*
N**
_output_shapes
:����������: 
�
sequential_1/dense_2/MatMulMatMul!sequential_1/dropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
�
sequential_1/dense_2/BiasAddBiasAddsequential_1/dense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:���������

b
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
[
num_inst/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
l
num_inst
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
�
num_inst/AssignAssignnum_instnum_inst/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@num_inst*
T0*
use_locking(
a
num_inst/readIdentitynum_inst*
T0*
_class
loc:@num_inst*
_output_shapes
: 
^
num_correct/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
o
num_correct
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
�
num_correct/AssignAssignnum_correctnum_correct/initial_value*
use_locking(*
T0*
_class
loc:@num_correct*
validate_shape(*
_output_shapes
: 
j
num_correct/readIdentitynum_correct*
_class
loc:@num_correct*
_output_shapes
: *
T0
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
e
ArgMaxArgMaxSoftmaxArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
g
ArgMax_1ArgMaxlabelArgMax_1/dimension*#
_output_shapes
:���������*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
S
ToFloatCastEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
X
SumSumToFloatConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
L
Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  �B
z
	AssignAdd	AssignAddnum_instConst_1*
use_locking( *
T0*
_class
loc:@num_inst*
_output_shapes
: 
~
AssignAdd_1	AssignAddnum_correctSum*
_class
loc:@num_correct*
_output_shapes
: *
T0*
use_locking( 
L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *    
�
AssignAssignnum_instConst_2*
_class
loc:@num_inst*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *    
�
Assign_1Assignnum_correctConst_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_correct
J
add/yConst*
dtype0*
_output_shapes
: *
valueB
 *���.
A
addAddnum_inst/readadd/y*
T0*
_output_shapes
: 
F
divRealDivnum_correct/readadd*
_output_shapes
: *
T0
L
div_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*'
_output_shapes
:���������
*
T0
a
softmax_cross_entropy_loss/RankConst*
dtype0*
_output_shapes
: *
value	B :
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
out_type0*
_output_shapes
:*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
T0*
out_type0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
�
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
_output_shapes
:*
N*

axis *
T0
o
%softmax_cross_entropy_loss/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
T0*
_output_shapes
:
}
*softmax_cross_entropy_loss/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
�
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
T0*
Tshape0*0
_output_shapes
:������������������
c
!softmax_cross_entropy_loss/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
T0*
out_type0*
_output_shapes
:
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*

axis *
_output_shapes
:*
T0*
N
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
T0*
_output_shapes
: 
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
valueB: *
_output_shapes
:*
dtype0
�
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
_output_shapes
:*
N*

axis *
T0
�
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
�
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
T0*#
_output_shapes
:���������*
Tshape0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
out_type0*
_output_shapes
:*
T0
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
�
1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes
: *
T0
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
�
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
T0*
_output_shapes
: 
�
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B : 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
_output_shapes
: *
dtype0
�
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0
�
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
�
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
T0*
_output_shapes
: 
�
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
u
%softmax_cross_entropy_loss/zeros_like	ZerosLike softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
�
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
N
PlaceholderPlaceholder*
_output_shapes
:*
dtype0*
shape: 
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
�
:gradients/softmax_cross_entropy_loss/value_grad/zeros_like	ZerosLikesoftmax_cross_entropy_loss/div*
T0*
_output_shapes
: 
�
6gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/Greatergradients/Fill:gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
T0*
_output_shapes
: 
�
8gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/Greater:gradients/softmax_cross_entropy_loss/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOp7^gradients/softmax_cross_entropy_loss/value_grad/Select9^gradients/softmax_cross_entropy_loss/value_grad/Select_1
�
Hgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_loss/value_grad/SelectA^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*
_output_shapes
: *I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select
�
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Cgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/div_grad/Shape5gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
1gradients/softmax_cross_entropy_loss/div_grad/SumSum5gradients/softmax_cross_entropy_loss/div_grad/RealDivCgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
5gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/div_grad/Sum3gradients/softmax_cross_entropy_loss/div_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
{
1gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv1gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDiv7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
1gradients/softmax_cross_entropy_loss/div_grad/mulMulHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
3gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum1gradients/softmax_cross_entropy_loss/div_grad/mulEgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: 
�
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
T0
�
;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like	ZerosLike$softmax_cross_entropy_loss/ones_like*
T0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Agradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss/Select_grad/Select:^gradients/softmax_cross_entropy_loss/Select_grad/Select_1
�
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select*
_output_shapes
: *
T0
�
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
�
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
�
>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 
�
4gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTile7gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
�
3gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
out_type0*
_output_shapes
:*
T0
�
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
�
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
_output_shapes
:*
out_type0*
T0
�
:gradients/softmax_cross_entropy_loss/num_present_grad/TileTile=gradients/softmax_cross_entropy_loss/num_present_grad/Reshape;gradients/softmax_cross_entropy_loss/num_present_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
_output_shapes
:*
out_type0
x
5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Cgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/Mul_grad/Shape5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1gradients/softmax_cross_entropy_loss/Mul_grad/mulMul2gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
1gradients/softmax_cross_entropy_loss/Mul_grad/SumSum1gradients/softmax_cross_entropy_loss/Mul_grad/mulCgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
�
3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_22gradients/softmax_cross_entropy_loss/Sum_grad/Tile*#
_output_shapes
:���������*
T0
�
3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Egradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
>gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape8^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:���������
�
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:*
out_type0
�
]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mulMul:gradients/softmax_cross_entropy_loss/num_present_grad/TileBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/Select:gradients/softmax_cross_entropy_loss/num_present_grad/Tile*
T0*#
_output_shapes
:���������
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1_gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
T0*#
_output_shapes
:���������*
Tshape0
�
Xgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOpP^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeR^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
�
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
_output_shapes
: *b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
T0
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*#
_output_shapes
:���������*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*
T0
�
Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
T0*
out_type0*
_output_shapes
:
�
;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/zeros_like	ZerosLike%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:������������������*
T0
�
Bgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradientPreventGradient%softmax_cross_entropy_loss/xentropy:1*
T0*0
_output_shapes
:������������������
�
Agradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
���������*
_output_shapes
: *
dtype0
�
=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDims;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*'
_output_shapes
:���������*
T0*

Tdim0
�
6gradients/softmax_cross_entropy_loss/xentropy_grad/mulMul=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDimsBgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradient*
T0*0
_output_shapes
:������������������
|
7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapediv_1*
_output_shapes
:*
out_type0*
T0
�
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
v
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
_
gradients/div_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/div_1_grad/RealDivRealDiv9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapediv_1/y*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/SumSumgradients/div_1_grad/RealDiv*gradients/div_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
o
gradients/div_1_grad/NegNegsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

~
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/RealDiv_2RealDivgradients/div_1_grad/RealDiv_1div_1/y*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/mulMul9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapegradients/div_1_grad/RealDiv_2*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/Sum_1Sumgradients/div_1_grad/mul,gradients/div_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_1_grad/Reshape_1Reshapegradients/div_1_grad/Sum_1gradients/div_1_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
m
%gradients/div_1_grad/tuple/group_depsNoOp^gradients/div_1_grad/Reshape^gradients/div_1_grad/Reshape_1
�
-gradients/div_1_grad/tuple/control_dependencyIdentitygradients/div_1_grad/Reshape&^gradients/div_1_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_1_grad/Reshape*
T0
�
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
_output_shapes
: *
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_1_grad/tuple/control_dependency*
_output_shapes
:
*
data_formatNHWC*
T0
�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_1_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_1_grad/Reshape*'
_output_shapes
:���������
*
T0
�
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad*
T0
�
1gradients/sequential_1/dense_2/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencydense_2/kernel/read*
transpose_b(*(
_output_shapes
:����������*
transpose_a( *
T0
�
3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1MatMul!sequential_1/dropout_2/cond/MergeDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	�
*
transpose_a(*
T0
�
;gradients/sequential_1/dense_2/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_2/MatMul_grad/MatMul4^gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	�
*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
�
:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradSwitchCgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
�
Agradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
�
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������
�
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
_output_shapes
:*
out_type0
Z
gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
t
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*(
_output_shapes
:����������
�
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros**
_output_shapes
:����������: *
N*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
out_type0*
_output_shapes
:*
T0
�
Lgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_2/cond/dropout/divKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*(
_output_shapes
:����������*
Tshape0
�
Ggradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape*(
_output_shapes
:����������
�
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*(
_output_shapes
:����������*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Lgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/NegNegsequential_1/dropout_2/cond/mul*(
_output_shapes
:����������*
T0
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Neg-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ggradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape*(
_output_shapes
:����������
�
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
T0*
out_type0*
_output_shapes
:
y
6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Dgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_2/cond/mul_grad/Shape6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/sequential_1/dropout_2/cond/mul_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
�
2gradients/sequential_1/dropout_2/cond/mul_grad/SumSum2gradients/sequential_1/dropout_2/cond/mul_grad/mulDgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
6gradients/sequential_1/dropout_2/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_2/cond/mul_grad/Sum4gradients/sequential_1/dropout_2/cond/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Mul(sequential_1/dropout_2/cond/mul/Switch:1Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
?gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape*(
_output_shapes
:����������
�
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
_output_shapes
: *
T0
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_2Shapegradients/Switch_1*
_output_shapes
:*
out_type0*
T0
\
gradients/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*(
_output_shapes
:����������
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1**
_output_shapes
:����������: *
N*
T0
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*
T0*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad*
N*(
_output_shapes
:����������
�
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*
T0*(
_output_shapes
:����������
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
_output_shapes	
:�*
data_formatNHWC*
T0
�
<gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_3/Relu_grad/ReluGrad8^gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad*(
_output_shapes
:����������
�
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
1gradients/sequential_1/dense_1/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
T0*)
_output_shapes
:�����������*
transpose_a( 
�
3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1MatMulsequential_1/flatten_1/ReshapeDgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *(
_output_shapes
:����������*
transpose_a(*
T0
�
;gradients/sequential_1/dense_1/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_1/MatMul_grad/MatMul4^gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*)
_output_shapes
:�����������*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*
T0
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*
T0*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
T0*/
_output_shapes
:���������@*
Tshape0
�
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape
�
Agradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@
�
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
T0
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
e
gradients/Shape_3Shapegradients/Switch_2:1*
_output_shapes
:*
out_type0*
T0
\
gradients/zeros_2/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0

gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*/
_output_shapes
:���������@*
T0
�
=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*
T0*
N*1
_output_shapes
:���������@: 
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0
�
Lgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_1/cond/dropout/divKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1*
T0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*/
_output_shapes
:���������@*
Tshape0*
T0
�
Ggradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape*/
_output_shapes
:���������@
�
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Lgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/NegNegsequential_1/dropout_1/cond/mul*
T0*/
_output_shapes
:���������@
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Neg-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2*/
_output_shapes
:���������@*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
Ggradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape*/
_output_shapes
:���������@
�
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
_output_shapes
:*
out_type0*
T0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Dgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_1/cond/mul_grad/Shape6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/sequential_1/dropout_1/cond/mul_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0
�
2gradients/sequential_1/dropout_1/cond/mul_grad/SumSum2gradients/sequential_1/dropout_1/cond/mul_grad/mulDgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Mul(sequential_1/dropout_1/cond/mul/Switch:1Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency*/
_output_shapes
:���������@*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
?gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_1/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape
�
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
gradients/Switch_3Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
c
gradients/Shape_4Shapegradients/Switch_3*
_output_shapes
:*
out_type0*
T0
\
gradients/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*/
_output_shapes
:���������@*
T0
�
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*1
_output_shapes
:���������@: *
N*
T0
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*
N*
T0*/
_output_shapes
:���������@*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*
T0*/
_output_shapes
:���������@
�
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@
�
=gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_2/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad
�
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
T0*
out_type0*
_output_shapes
:
�
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
strides
*
T0*
paddingVALID
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"      @   @   
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:@@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
�
Agradients/sequential_1/conv2d_2/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������@*
T0
�
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*&
_output_shapes
:@@*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
_output_shapes
:@*
T0*
data_formatNHWC
�
=gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_1/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_1/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad*/
_output_shapes
:���������@
�
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
_output_shapes
:*
out_type0*
T0
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4������������������������������������
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*%
valueB"         @   *
dtype0*
_output_shapes
:
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
�
Agradients/sequential_1/conv2d_1/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������*
T0
�
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@
�
beta1_power/initial_valueConst*
valueB
 *fff?*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0
�
beta1_power
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
shape: *
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
n
beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
�
beta2_power/initial_valueConst*
valueB
 *w�?*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0
�
beta2_power
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*
shared_name *
_output_shapes
: *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
n
beta2_power/readIdentitybeta2_power*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
j
zerosConst*
dtype0*&
_output_shapes
:@*%
valueB@*    
�
conv2d_1/kernel/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:@*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
l
zeros_1Const*
dtype0*&
_output_shapes
:@*%
valueB@*    
�
conv2d_1/kernel/Adam_1
VariableV2*
shared_name *
shape:@*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
dtype0*
	container 
�
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0
T
zeros_2Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_1/bias/Adam
VariableV2*
	container *
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
shape:@*
shared_name 
�
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
~
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
T
zeros_3Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
conv2d_1/bias/Adam_1
VariableV2*
	container *
dtype0* 
_class
loc:@conv2d_1/bias*
shared_name *
_output_shapes
:@*
shape:@
�
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
l
zeros_4Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
�
conv2d_2/kernel/Adam
VariableV2*
shared_name *
shape:@@*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
dtype0*
	container 
�
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
l
zeros_5Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
�
conv2d_2/kernel/Adam_1
VariableV2*&
_output_shapes
:@@*
dtype0*
shape:@@*
	container *"
_class
loc:@conv2d_2/kernel*
shared_name 
�
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
T
zeros_6Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
conv2d_2/bias/Adam
VariableV2*
shape:@*
_output_shapes
:@*
shared_name * 
_class
loc:@conv2d_2/bias*
dtype0*
	container 
�
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
~
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
T
zeros_7Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_2/bias/Adam_1
VariableV2* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
shape:@*
dtype0*
shared_name *
	container 
�
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
b
zeros_8Const*
dtype0*!
_output_shapes
:���* 
valueB���*    
�
dense_1/kernel/Adam
VariableV2*
shared_name *
shape:���*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
dtype0*
	container 
�
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
b
zeros_9Const* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
dense_1/kernel/Adam_1
VariableV2*
shape:���*!
_output_shapes
:���*
shared_name *!
_class
loc:@dense_1/kernel*
dtype0*
	container 
�
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1zeros_9*!
_output_shapes
:���*
validate_shape(*!
_class
loc:@dense_1/kernel*
T0*
use_locking(
�
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
W
zeros_10Const*
valueB�*    *
_output_shapes	
:�*
dtype0
�
dense_1/bias/Adam
VariableV2*
_output_shapes	
:�*
dtype0*
shape:�*
	container *
_class
loc:@dense_1/bias*
shared_name 
�
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
|
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
W
zeros_11Const*
valueB�*    *
_output_shapes	
:�*
dtype0
�
dense_1/bias/Adam_1
VariableV2*
_output_shapes	
:�*
dtype0*
shape:�*
	container *
_class
loc:@dense_1/bias*
shared_name 
�
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
�
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
_
zeros_12Const*
valueB	�
*    *
dtype0*
_output_shapes
:	�

�
dense_2/kernel/Adam
VariableV2*
	container *
dtype0*!
_class
loc:@dense_2/kernel*
shared_name *
_output_shapes
:	�
*
shape:	�

�
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
�
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

_
zeros_13Const*
dtype0*
_output_shapes
:	�
*
valueB	�
*    
�
dense_2/kernel/Adam_1
VariableV2*
shape:	�
*
_output_shapes
:	�
*
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container 
�
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1zeros_13*
_output_shapes
:	�
*
validate_shape(*!
_class
loc:@dense_2/kernel*
T0*
use_locking(
�
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

U
zeros_14Const*
_output_shapes
:
*
dtype0*
valueB
*    
�
dense_2/bias/Adam
VariableV2*
	container *
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
:
*
shape:
*
shared_name 
�
dense_2/bias/Adam/AssignAssigndense_2/bias/Adamzeros_14*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
{
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
U
zeros_15Const*
dtype0*
_output_shapes
:
*
valueB
*    
�
dense_2/bias/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@dense_2/bias*
shared_name *
_output_shapes
:
*
shape:

�
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
_output_shapes
:
*
validate_shape(*
_class
loc:@dense_2/bias*
T0*
use_locking(

dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
_output_shapes
:
*
_class
loc:@dense_2/bias*
T0
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w�+2
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
use_locking( 
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0*
use_locking( 
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
use_locking( 
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0*
use_locking( 
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
use_locking( 
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0*
use_locking( 
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

�
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

�
Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking( 
�
AdamNoOp&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
N
	loss/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Bloss
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"�^N���     ����	�#RzYc�AJ��
�%�$
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
:
Greater
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
5
PreventGradient

input"T
output"T"	
Ttype
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirty��
^
dataPlaceholder*
shape: *
dtype0*/
_output_shapes
:���������
W
labelPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������

h
conv2d_1_inputPlaceholder*/
_output_shapes
:���������*
dtype0*
shape: 
v
conv2d_1/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
`
conv2d_1/random_uniform/minConst*
valueB
 *�x�*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *�x=*
_output_shapes
: *
dtype0
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2���*
T0*
seed���)*
dtype0
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 
�
conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
:@*
T0
�
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
:@*
T0
�
conv2d_1/kernel
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
�
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0
[
conv2d_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_1/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
�
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
s
conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
�
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC
e
activation_1/ReluReluconv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
v
conv2d_2/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
`
conv2d_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�\1�
`
conv2d_2/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�\1=
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
dtype0*
seed���)*
T0*&
_output_shapes
:@@*
seed2���
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0
�
conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
:@@
�
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
:@@
�
conv2d_2/kernel
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
dtype0*
shared_name 
�
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
[
conv2d_2/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
y
conv2d_2/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
�
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_2/bias/readIdentityconv2d_2/bias*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0
s
conv2d_2/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
s
"conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
use_cudnn_on_gpu(*/
_output_shapes
:���������@*
data_formatNHWC*
strides
*
T0*
paddingVALID
�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC
e
activation_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:���������@
a
dropout_1/keras_learning_phasePlaceholder*
_output_shapes
:*
dtype0
*
shape: 
�
dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
:*
T0

e
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*
T0*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@
�
dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2��*
dtype0*
T0*
seed���)
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*/
_output_shapes
:���������@*
T0
�
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*/
_output_shapes
:���������@*
T0
{
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
N*
T0*1
_output_shapes
:���������@: 
c
flatten_1/ShapeShapedropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
g
flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
:*
end_mask*
Index0*
T0*
shrink_axis_mask *
new_axis_mask 
Y
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
\
flatten_1/stack/0Const*
valueB :
���������*
_output_shapes
: *
dtype0
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
_output_shapes
:*
N*

axis *
T0
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
T0*0
_output_shapes
:������������������*
Tshape0
m
dense_1/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB" d  �   
_
dense_1/random_uniform/minConst*
valueB
 *�3z�*
_output_shapes
: *
dtype0
_
dense_1/random_uniform/maxConst*
valueB
 *�3z<*
_output_shapes
: *
dtype0
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2���*
dtype0*
T0*
seed���)
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*!
_output_shapes
:���
�
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*!
_output_shapes
:���*
T0
�
dense_1/kernel
VariableV2*!
_output_shapes
:���*
	container *
dtype0*
shared_name *
shape:���
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
~
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
valueB�*    *
_output_shapes	
:�*
dtype0
z
dense_1/bias
VariableV2*
_output_shapes	
:�*
	container *
shape:�*
dtype0*
shared_name 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *(
_output_shapes
:����������*
transpose_a( *
T0
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
]
activation_3/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:����������
�
dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
:
e
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��.*
dtype0*
T0*
seed���)
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:����������
t
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
�
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul**
_output_shapes
:����������: *
T0*
N
m
dense_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"�   
   
_
dense_2/random_uniform/minConst*
valueB
 *̈́U�*
_output_shapes
: *
dtype0
_
dense_2/random_uniform/maxConst*
valueB
 *̈́U>*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes
:	�
*
seed2���
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes
:	�


dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	�
*
T0
�
dense_2/kernel
VariableV2*
shared_name *
dtype0*
shape:	�
*
_output_shapes
:	�
*
	container 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0*
validate_shape(*
use_locking(
|
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0
Z
dense_2/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*    
x
dense_2/bias
VariableV2*
_output_shapes
:
*
	container *
dtype0*
shared_name *
shape:

�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

�
dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:���������

�
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
�
'sequential_1/conv2d_1/convolution/ShapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������@
�
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
�
'sequential_1/conv2d_2/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
�
/sequential_1/conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC

sequential_1/activation_2/ReluRelusequential_1/conv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
�
"sequential_1/dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

y
$sequential_1/dropout_1/cond/switch_tIdentity$sequential_1/dropout_1/cond/Switch:1*
T0
*
_output_shapes
:
w
$sequential_1/dropout_1/cond/switch_fIdentity"sequential_1/dropout_1/cond/Switch*
T0
*
_output_shapes
:
r
#sequential_1/dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

�
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
&sequential_1/dropout_1/cond/mul/SwitchSwitchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu*
T0
�
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@
�
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2��H*
dtype0*
T0*
seed���)
�
6sequential_1/dropout_1/cond/dropout/random_uniform/subSub6sequential_1/dropout_1/cond/dropout/random_uniform/max6sequential_1/dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
6sequential_1/dropout_1/cond/dropout/random_uniform/mulMul@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_1/cond/dropout/random_uniform/sub*/
_output_shapes
:���������@*
T0
�
2sequential_1/dropout_1/cond/dropout/random_uniformAdd6sequential_1/dropout_1/cond/dropout/random_uniform/mul6sequential_1/dropout_1/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:���������@
�
'sequential_1/dropout_1/cond/dropout/addAdd-sequential_1/dropout_1/cond/dropout/keep_prob2sequential_1/dropout_1/cond/dropout/random_uniform*/
_output_shapes
:���������@*
T0
�
)sequential_1/dropout_1/cond/dropout/FloorFloor'sequential_1/dropout_1/cond/dropout/add*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/divRealDivsequential_1/dropout_1/cond/mul-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/mulMul'sequential_1/dropout_1/cond/dropout/div)sequential_1/dropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
$sequential_1/dropout_1/cond/Switch_1Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu
�
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
N*
T0
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
t
*sequential_1/flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
v
,sequential_1/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
_output_shapes
:*
end_mask*
new_axis_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask *
T0*
Index0
f
sequential_1/flatten_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
sequential_1/flatten_1/ProdProd$sequential_1/flatten_1/strided_slicesequential_1/flatten_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
sequential_1/flatten_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:
�
sequential_1/flatten_1/ReshapeReshape!sequential_1/dropout_1/cond/Mergesequential_1/flatten_1/stack*0
_output_shapes
:������������������*
Tshape0*
T0
�
sequential_1/dense_1/MatMulMatMulsequential_1/flatten_1/Reshapedense_1/kernel/read*
transpose_b( *(
_output_shapes
:����������*
transpose_a( *
T0
�
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
w
sequential_1/activation_3/ReluRelusequential_1/dense_1/BiasAdd*
T0*(
_output_shapes
:����������
�
"sequential_1/dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

y
$sequential_1/dropout_2/cond/switch_tIdentity$sequential_1/dropout_2/cond/Switch:1*
T0
*
_output_shapes
:
w
$sequential_1/dropout_2/cond/switch_fIdentity"sequential_1/dropout_2/cond/Switch*
T0
*
_output_shapes
:
r
#sequential_1/dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
�
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu
�
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
�
-sequential_1/dropout_2/cond/dropout/keep_probConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/minConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6sequential_1/dropout_2/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
6sequential_1/dropout_2/cond/dropout/random_uniform/mulMul@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_2/cond/dropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
2sequential_1/dropout_2/cond/dropout/random_uniformAdd6sequential_1/dropout_2/cond/dropout/random_uniform/mul6sequential_1/dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/addAdd-sequential_1/dropout_2/cond/dropout/keep_prob2sequential_1/dropout_2/cond/dropout/random_uniform*(
_output_shapes
:����������*
T0
�
)sequential_1/dropout_2/cond/dropout/FloorFloor'sequential_1/dropout_2/cond/dropout/add*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/divRealDivsequential_1/dropout_2/cond/mul-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/mulMul'sequential_1/dropout_2/cond/dropout/div)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������*
T0
�
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul*
N*
T0**
_output_shapes
:����������: 
�
sequential_1/dense_2/MatMulMatMul!sequential_1/dropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
�
sequential_1/dense_2/BiasAddBiasAddsequential_1/dense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:���������

b
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
[
num_inst/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
l
num_inst
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
num_inst/AssignAssignnum_instnum_inst/initial_value*
use_locking(*
T0*
_class
loc:@num_inst*
validate_shape(*
_output_shapes
: 
a
num_inst/readIdentitynum_inst*
_output_shapes
: *
_class
loc:@num_inst*
T0
^
num_correct/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
num_correct
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
num_correct/AssignAssignnum_correctnum_correct/initial_value*
use_locking(*
T0*
_class
loc:@num_correct*
validate_shape(*
_output_shapes
: 
j
num_correct/readIdentitynum_correct*
_output_shapes
: *
_class
loc:@num_correct*
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
e
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
g
ArgMax_1ArgMaxlabelArgMax_1/dimension*#
_output_shapes
:���������*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
S
ToFloatCastEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
O
ConstConst*
valueB: *
_output_shapes
:*
dtype0
X
SumSumToFloatConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
L
Const_1Const*
valueB
 *  �B*
dtype0*
_output_shapes
: 
z
	AssignAdd	AssignAddnum_instConst_1*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@num_inst
~
AssignAdd_1	AssignAddnum_correctSum*
_class
loc:@num_correct*
_output_shapes
: *
T0*
use_locking( 
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
�
AssignAssignnum_instConst_2*
_output_shapes
: *
validate_shape(*
_class
loc:@num_inst*
T0*
use_locking(
L
Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Assign_1Assignnum_correctConst_3*
_class
loc:@num_correct*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
J
add/yConst*
valueB
 *���.*
dtype0*
_output_shapes
: 
A
addAddnum_inst/readadd/y*
_output_shapes
: *
T0
F
divRealDivnum_correct/readadd*
T0*
_output_shapes
: 
L
div_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*'
_output_shapes
:���������
*
T0
a
softmax_cross_entropy_loss/RankConst*
_output_shapes
: *
dtype0*
value	B :
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
out_type0*
_output_shapes
:*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
_output_shapes
:*
out_type0*
T0
b
 softmax_cross_entropy_loss/Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
�
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
T0*

axis *
N*
_output_shapes
:
o
%softmax_cross_entropy_loss/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
_output_shapes
:*
Index0*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*0
_output_shapes
:������������������*
Tshape0*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
out_type0*
_output_shapes
:*
T0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
_output_shapes
:*
N*

axis *
T0
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
d
"softmax_cross_entropy_loss/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
�
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
T0*
_output_shapes
: 
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*

axis *
_output_shapes
:*
T0*
N
�
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
�
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
T0*
Tshape0*#
_output_shapes
:���������
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
_output_shapes
: *
dtype0
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
out_type0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
T0*
_output_shapes
: 
�
1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes
: *
T0
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
T0*
_output_shapes
: 
�
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
�
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
_output_shapes
: *
dtype0
�
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0
�
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
T0*
_output_shapes
: 
�
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0
�
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
u
%softmax_cross_entropy_loss/zeros_like	ZerosLike softmax_cross_entropy_loss/Sum_1*
T0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
N
PlaceholderPlaceholder*
_output_shapes
:*
dtype0*
shape: 
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
�
:gradients/softmax_cross_entropy_loss/value_grad/zeros_like	ZerosLikesoftmax_cross_entropy_loss/div*
T0*
_output_shapes
: 
�
6gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/Greatergradients/Fill:gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
�
8gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/Greater:gradients/softmax_cross_entropy_loss/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOp7^gradients/softmax_cross_entropy_loss/value_grad/Select9^gradients/softmax_cross_entropy_loss/value_grad/Select_1
�
Hgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_loss/value_grad/SelectA^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*
_output_shapes
: *I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select
�
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: 
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Cgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/div_grad/Shape5gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
5gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
�
1gradients/softmax_cross_entropy_loss/div_grad/SumSum5gradients/softmax_cross_entropy_loss/div_grad/RealDivCgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
5gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/div_grad/Sum3gradients/softmax_cross_entropy_loss/div_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
{
1gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv1gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDiv7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
�
1gradients/softmax_cross_entropy_loss/div_grad/mulMulHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
3gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum1gradients/softmax_cross_entropy_loss/div_grad/mulEgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: 
�
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0
�
;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like	ZerosLike$softmax_cross_entropy_loss/ones_like*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Agradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss/Select_grad/Select:^gradients/softmax_cross_entropy_loss/Select_grad/Select_1
�
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select
�
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
T0
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB 
�
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
T0*
_output_shapes
: *
Tshape0
�
>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
_output_shapes
: *
dtype0*
valueB 
�
4gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTile7gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
�
;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
�
3gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
T0*
_output_shapes
:*
out_type0
�
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
_output_shapes
:*
out_type0*
T0
�
:gradients/softmax_cross_entropy_loss/num_present_grad/TileTile=gradients/softmax_cross_entropy_loss/num_present_grad/Reshape;gradients/softmax_cross_entropy_loss/num_present_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
_output_shapes
:*
out_type0
x
5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Cgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/Mul_grad/Shape5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1gradients/softmax_cross_entropy_loss/Mul_grad/mulMul2gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
1gradients/softmax_cross_entropy_loss/Mul_grad/SumSum1gradients/softmax_cross_entropy_loss/Mul_grad/mulCgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
�
3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_22gradients/softmax_cross_entropy_loss/Sum_grad/Tile*
T0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Egradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
>gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape8^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:���������*
T0
�
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: 
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:*
out_type0
�
]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mulMul:gradients/softmax_cross_entropy_loss/num_present_grad/TileBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
T0*
_output_shapes
: *
Tshape0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/Select:gradients/softmax_cross_entropy_loss/num_present_grad/Tile*#
_output_shapes
:���������*
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1_gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
�
Xgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOpP^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeR^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
�
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
_output_shapes
: *b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
T0
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������*
T0
�
Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
out_type0*
_output_shapes
:*
T0
�
;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
�
gradients/zeros_like	ZerosLike%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:������������������*
T0
�
Bgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradientPreventGradient%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:������������������*
T0
�
Agradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
���������*
_output_shapes
: *
dtype0
�
=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDims;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
6gradients/softmax_cross_entropy_loss/xentropy_grad/mulMul=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDimsBgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradient*
T0*0
_output_shapes
:������������������
|
7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapediv_1*
_output_shapes
:*
out_type0*
T0
�
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
v
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
_
gradients/div_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/div_1_grad/RealDivRealDiv9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapediv_1/y*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/SumSumgradients/div_1_grad/RealDiv*gradients/div_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

o
gradients/div_1_grad/NegNegsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
~
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/RealDiv_2RealDivgradients/div_1_grad/RealDiv_1div_1/y*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/mulMul9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapegradients/div_1_grad/RealDiv_2*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/Sum_1Sumgradients/div_1_grad/mul,gradients/div_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_1_grad/Reshape_1Reshapegradients/div_1_grad/Sum_1gradients/div_1_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
m
%gradients/div_1_grad/tuple/group_depsNoOp^gradients/div_1_grad/Reshape^gradients/div_1_grad/Reshape_1
�
-gradients/div_1_grad/tuple/control_dependencyIdentitygradients/div_1_grad/Reshape&^gradients/div_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_1_grad/Reshape*'
_output_shapes
:���������

�
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*
_output_shapes
: *1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_1_grad/tuple/control_dependency*
_output_shapes
:
*
T0*
data_formatNHWC
�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_1_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_1_grad/Reshape*'
_output_shapes
:���������

�
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad*
T0
�
1gradients/sequential_1/dense_2/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencydense_2/kernel/read*
transpose_b(*(
_output_shapes
:����������*
transpose_a( *
T0
�
3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1MatMul!sequential_1/dropout_2/cond/MergeDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�
*
transpose_a(
�
;gradients/sequential_1/dense_2/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_2/MatMul_grad/MatMul4^gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	�

�
:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradSwitchCgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
�
Agradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
�
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
�
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
_output_shapes
:*
out_type0
Z
gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
t
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*(
_output_shapes
:����������*
T0
�
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros*
T0*
N**
_output_shapes
:����������: 
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
out_type0*
_output_shapes
:*
T0
�
Lgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_2/cond/dropout/divKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Ggradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape*
T0
�
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Lgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/NegNegsequential_1/dropout_2/cond/mul*
T0*(
_output_shapes
:����������
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Neg-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
Ggradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape*(
_output_shapes
:����������
�
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
_output_shapes
:*
out_type0*
T0
y
6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Dgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_2/cond/mul_grad/Shape6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/sequential_1/dropout_2/cond/mul_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
�
2gradients/sequential_1/dropout_2/cond/mul_grad/SumSum2gradients/sequential_1/dropout_2/cond/mul_grad/mulDgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
6gradients/sequential_1/dropout_2/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_2/cond/mul_grad/Sum4gradients/sequential_1/dropout_2/cond/mul_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Mul(sequential_1/dropout_2/cond/mul/Switch:1Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
?gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape*
T0
�
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
T0
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
_output_shapes
:*
out_type0
\
gradients/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*(
_output_shapes
:����������*
T0
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1*
N*
T0**
_output_shapes
:����������: 
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*
N*
T0*(
_output_shapes
:����������*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad
�
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*
T0*(
_output_shapes
:����������
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
<gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_3/Relu_grad/ReluGrad8^gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad*(
_output_shapes
:����������
�
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
T0
�
1gradients/sequential_1/dense_1/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*)
_output_shapes
:�����������*
transpose_a( *
T0
�
3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1MatMulsequential_1/flatten_1/ReshapeDgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a(
�
;gradients/sequential_1/dense_1/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_1/MatMul_grad/MatMul4^gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*
T0*)
_output_shapes
:�����������*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*
T0
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
_output_shapes
:*
out_type0*
T0
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*J
_output_shapes8
6:���������@:���������@
�
Agradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@
�
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_2/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0

gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*/
_output_shapes
:���������@*
T0
�
=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*
T0*
N*1
_output_shapes
:���������@: 
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
_output_shapes
:*
out_type0*
T0
�
Lgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_1/cond/dropout/divKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1*/
_output_shapes
:���������@*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:���������@
�
Ggradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape
�
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1*/
_output_shapes
:���������@*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Lgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/NegNegsequential_1/dropout_1/cond/mul*
T0*/
_output_shapes
:���������@
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Neg-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2*/
_output_shapes
:���������@*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
Ggradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape*
T0
�
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
out_type0*
_output_shapes
:*
T0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Dgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_1/cond/mul_grad/Shape6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/sequential_1/dropout_1/cond/mul_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0
�
2gradients/sequential_1/dropout_1/cond/mul_grad/SumSum2gradients/sequential_1/dropout_1/cond/mul_grad/mulDgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*
T0*/
_output_shapes
:���������@*
Tshape0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Mul(sequential_1/dropout_1/cond/mul/Switch:1Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency*/
_output_shapes
:���������@*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
?gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_1/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape*
T0
�
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1*
T0
�
gradients/Switch_3Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*/
_output_shapes
:���������@
�
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*
N*
T0*1
_output_shapes
:���������@: 
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*
T0*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad*
N*/
_output_shapes
:���������@
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*
T0*/
_output_shapes
:���������@
�
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
=gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_2/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
T0
�
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
�
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
T0*
out_type0*
_output_shapes
:
�
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4������������������������������������
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*%
valueB"      @   @   *
_output_shapes
:*
dtype0
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:@@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
�
Agradients/sequential_1/conv2d_2/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������@*
T0
�
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*&
_output_shapes
:@@*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*
T0*/
_output_shapes
:���������@
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
_output_shapes
:@*
T0*
data_formatNHWC
�
=gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_1/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_1/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad
�
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
T0*
_output_shapes
:*
out_type0
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*%
valueB"         @   *
_output_shapes
:*
dtype0
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*&
_output_shapes
:@*
data_formatNHWC*
strides
*
T0*
paddingVALID
�
Agradients/sequential_1/conv2d_1/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*/
_output_shapes
:���������*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput*
T0
�
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*
T0*&
_output_shapes
:@*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
beta1_power/initial_valueConst*
valueB
 *fff?*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0
�
beta1_power
VariableV2*
shape: *
_output_shapes
: *
shared_name *"
_class
loc:@conv2d_1/kernel*
dtype0*
	container 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
n
beta1_power/readIdentitybeta1_power*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
�
beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *w�?*"
_class
loc:@conv2d_1/kernel
�
beta2_power
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
n
beta2_power/readIdentitybeta2_power*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
j
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    
�
conv2d_1/kernel/Adam
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
shape:@*
shared_name 
�
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0
l
zeros_1Const*
dtype0*&
_output_shapes
:@*%
valueB@*    
�
conv2d_1/kernel/Adam_1
VariableV2*&
_output_shapes
:@*
dtype0*
shape:@*
	container *"
_class
loc:@conv2d_1/kernel*
shared_name 
�
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
T
zeros_2Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_1/bias/Adam
VariableV2*
shape:@*
_output_shapes
:@*
shared_name * 
_class
loc:@conv2d_1/bias*
dtype0*
	container 
�
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
~
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
T
zeros_3Const*
_output_shapes
:@*
dtype0*
valueB@*    
�
conv2d_1/bias/Adam_1
VariableV2*
	container *
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
shape:@*
shared_name 
�
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
l
zeros_4Const*%
valueB@@*    *&
_output_shapes
:@@*
dtype0
�
conv2d_2/kernel/Adam
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_2/kernel*
shared_name *&
_output_shapes
:@@*
shape:@@
�
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking(
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
l
zeros_5Const*%
valueB@@*    *&
_output_shapes
:@@*
dtype0
�
conv2d_2/kernel/Adam_1
VariableV2*
shared_name *
shape:@@*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
dtype0*
	container 
�
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
T
zeros_6Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_2/bias/Adam
VariableV2*
shape:@*
_output_shapes
:@*
shared_name * 
_class
loc:@conv2d_2/bias*
dtype0*
	container 
�
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
~
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0
T
zeros_7Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_2/bias/Adam_1
VariableV2* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
shape:@*
dtype0*
shared_name *
	container 
�
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
b
zeros_8Const* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
dense_1/kernel/Adam
VariableV2*
	container *
dtype0*!
_class
loc:@dense_1/kernel*
shared_name *!
_output_shapes
:���*
shape:���
�
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0
b
zeros_9Const* 
valueB���*    *!
_output_shapes
:���*
dtype0
�
dense_1/kernel/Adam_1
VariableV2*
	container *
dtype0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
shape:���*
shared_name 
�
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1zeros_9*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*!
_output_shapes
:���
�
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
W
zeros_10Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_1/bias/Adam
VariableV2*
shared_name *
shape:�*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
dtype0*
	container 
�
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
|
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0
W
zeros_11Const*
valueB�*    *
_output_shapes	
:�*
dtype0
�
dense_1/bias/Adam_1
VariableV2*
_output_shapes	
:�*
dtype0*
shape:�*
	container *
_class
loc:@dense_1/bias*
shared_name 
�
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
_output_shapes	
:�*
validate_shape(*
_class
loc:@dense_1/bias*
T0*
use_locking(
�
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
_
zeros_12Const*
dtype0*
_output_shapes
:	�
*
valueB	�
*    
�
dense_2/kernel/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:	�
*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
�
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
�
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0
_
zeros_13Const*
_output_shapes
:	�
*
dtype0*
valueB	�
*    
�
dense_2/kernel/Adam_1
VariableV2*
shared_name *!
_class
loc:@dense_2/kernel*
	container *
shape:	�
*
dtype0*
_output_shapes
:	�

�
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1zeros_13*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
�
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

U
zeros_14Const*
valueB
*    *
_output_shapes
:
*
dtype0
�
dense_2/bias/Adam
VariableV2*
_class
loc:@dense_2/bias*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
�
dense_2/bias/Adam/AssignAssigndense_2/bias/Adamzeros_14*
_output_shapes
:
*
validate_shape(*
_class
loc:@dense_2/bias*
T0*
use_locking(
{
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
U
zeros_15Const*
_output_shapes
:
*
dtype0*
valueB
*    
�
dense_2/bias/Adam_1
VariableV2*
_class
loc:@dense_2/bias*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
�
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:


dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
_output_shapes
:
*
_class
loc:@dense_2/bias*
T0
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *w�?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *w�+2*
_output_shapes
: *
dtype0
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0*
use_locking( 
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
use_locking( 
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
�
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:
*
_class
loc:@dense_2/bias*
T0*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
�
AdamNoOp&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N""0
losses&
$
"softmax_cross_entropy_loss/value:0"
	summaries


loss:0"�
trainable_variables��
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
C
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:0
=
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0"
train_op

Adam"�&
cond_context�&�&
�
dropout_1/cond/cond_textdropout_1/cond/pred_id:0dropout_1/cond/switch_t:0 *�
activation_2/Relu:0
dropout_1/cond/dropout/Floor:0
dropout_1/cond/dropout/Shape:0
dropout_1/cond/dropout/add:0
dropout_1/cond/dropout/div:0
"dropout_1/cond/dropout/keep_prob:0
dropout_1/cond/dropout/mul:0
5dropout_1/cond/dropout/random_uniform/RandomUniform:0
+dropout_1/cond/dropout/random_uniform/max:0
+dropout_1/cond/dropout/random_uniform/min:0
+dropout_1/cond/dropout/random_uniform/mul:0
+dropout_1/cond/dropout/random_uniform/sub:0
'dropout_1/cond/dropout/random_uniform:0
dropout_1/cond/mul/Switch:1
dropout_1/cond/mul/y:0
dropout_1/cond/mul:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_t:02
activation_2/Relu:0dropout_1/cond/mul/Switch:1
�
dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*�
activation_2/Relu:0
dropout_1/cond/Switch_1:0
dropout_1/cond/Switch_1:1
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:00
activation_2/Relu:0dropout_1/cond/Switch_1:0
�
dropout_2/cond/cond_textdropout_2/cond/pred_id:0dropout_2/cond/switch_t:0 *�
activation_3/Relu:0
dropout_2/cond/dropout/Floor:0
dropout_2/cond/dropout/Shape:0
dropout_2/cond/dropout/add:0
dropout_2/cond/dropout/div:0
"dropout_2/cond/dropout/keep_prob:0
dropout_2/cond/dropout/mul:0
5dropout_2/cond/dropout/random_uniform/RandomUniform:0
+dropout_2/cond/dropout/random_uniform/max:0
+dropout_2/cond/dropout/random_uniform/min:0
+dropout_2/cond/dropout/random_uniform/mul:0
+dropout_2/cond/dropout/random_uniform/sub:0
'dropout_2/cond/dropout/random_uniform:0
dropout_2/cond/mul/Switch:1
dropout_2/cond/mul/y:0
dropout_2/cond/mul:0
dropout_2/cond/pred_id:0
dropout_2/cond/switch_t:02
activation_3/Relu:0dropout_2/cond/mul/Switch:1
�
dropout_2/cond/cond_text_1dropout_2/cond/pred_id:0dropout_2/cond/switch_f:0*�
activation_3/Relu:0
dropout_2/cond/Switch_1:0
dropout_2/cond/Switch_1:1
dropout_2/cond/pred_id:0
dropout_2/cond/switch_f:00
activation_3/Relu:0dropout_2/cond/Switch_1:0
�
%sequential_1/dropout_1/cond/cond_text%sequential_1/dropout_1/cond/pred_id:0&sequential_1/dropout_1/cond/switch_t:0 *�
 sequential_1/activation_2/Relu:0
+sequential_1/dropout_1/cond/dropout/Floor:0
+sequential_1/dropout_1/cond/dropout/Shape:0
)sequential_1/dropout_1/cond/dropout/add:0
)sequential_1/dropout_1/cond/dropout/div:0
/sequential_1/dropout_1/cond/dropout/keep_prob:0
)sequential_1/dropout_1/cond/dropout/mul:0
Bsequential_1/dropout_1/cond/dropout/random_uniform/RandomUniform:0
8sequential_1/dropout_1/cond/dropout/random_uniform/max:0
8sequential_1/dropout_1/cond/dropout/random_uniform/min:0
8sequential_1/dropout_1/cond/dropout/random_uniform/mul:0
8sequential_1/dropout_1/cond/dropout/random_uniform/sub:0
4sequential_1/dropout_1/cond/dropout/random_uniform:0
(sequential_1/dropout_1/cond/mul/Switch:1
#sequential_1/dropout_1/cond/mul/y:0
!sequential_1/dropout_1/cond/mul:0
%sequential_1/dropout_1/cond/pred_id:0
&sequential_1/dropout_1/cond/switch_t:0L
 sequential_1/activation_2/Relu:0(sequential_1/dropout_1/cond/mul/Switch:1
�
'sequential_1/dropout_1/cond/cond_text_1%sequential_1/dropout_1/cond/pred_id:0&sequential_1/dropout_1/cond/switch_f:0*�
 sequential_1/activation_2/Relu:0
&sequential_1/dropout_1/cond/Switch_1:0
&sequential_1/dropout_1/cond/Switch_1:1
%sequential_1/dropout_1/cond/pred_id:0
&sequential_1/dropout_1/cond/switch_f:0J
 sequential_1/activation_2/Relu:0&sequential_1/dropout_1/cond/Switch_1:0
�
%sequential_1/dropout_2/cond/cond_text%sequential_1/dropout_2/cond/pred_id:0&sequential_1/dropout_2/cond/switch_t:0 *�
 sequential_1/activation_3/Relu:0
+sequential_1/dropout_2/cond/dropout/Floor:0
+sequential_1/dropout_2/cond/dropout/Shape:0
)sequential_1/dropout_2/cond/dropout/add:0
)sequential_1/dropout_2/cond/dropout/div:0
/sequential_1/dropout_2/cond/dropout/keep_prob:0
)sequential_1/dropout_2/cond/dropout/mul:0
Bsequential_1/dropout_2/cond/dropout/random_uniform/RandomUniform:0
8sequential_1/dropout_2/cond/dropout/random_uniform/max:0
8sequential_1/dropout_2/cond/dropout/random_uniform/min:0
8sequential_1/dropout_2/cond/dropout/random_uniform/mul:0
8sequential_1/dropout_2/cond/dropout/random_uniform/sub:0
4sequential_1/dropout_2/cond/dropout/random_uniform:0
(sequential_1/dropout_2/cond/mul/Switch:1
#sequential_1/dropout_2/cond/mul/y:0
!sequential_1/dropout_2/cond/mul:0
%sequential_1/dropout_2/cond/pred_id:0
&sequential_1/dropout_2/cond/switch_t:0L
 sequential_1/activation_3/Relu:0(sequential_1/dropout_2/cond/mul/Switch:1
�
'sequential_1/dropout_2/cond/cond_text_1%sequential_1/dropout_2/cond/pred_id:0&sequential_1/dropout_2/cond/switch_f:0*�
 sequential_1/activation_3/Relu:0
&sequential_1/dropout_2/cond/Switch_1:0
&sequential_1/dropout_2/cond/Switch_1:1
%sequential_1/dropout_2/cond/pred_id:0
&sequential_1/dropout_2/cond/switch_f:0J
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0"�
	variables��
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
C
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:0
=
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0
.

num_inst:0num_inst/Assignnum_inst/read:0
7
num_correct:0num_correct/Assignnum_correct/read:0
7
beta1_power:0beta1_power/Assignbeta1_power/read:0
7
beta2_power:0beta2_power/Assignbeta2_power/read:0
R
conv2d_1/kernel/Adam:0conv2d_1/kernel/Adam/Assignconv2d_1/kernel/Adam/read:0
X
conv2d_1/kernel/Adam_1:0conv2d_1/kernel/Adam_1/Assignconv2d_1/kernel/Adam_1/read:0
L
conv2d_1/bias/Adam:0conv2d_1/bias/Adam/Assignconv2d_1/bias/Adam/read:0
R
conv2d_1/bias/Adam_1:0conv2d_1/bias/Adam_1/Assignconv2d_1/bias/Adam_1/read:0
R
conv2d_2/kernel/Adam:0conv2d_2/kernel/Adam/Assignconv2d_2/kernel/Adam/read:0
X
conv2d_2/kernel/Adam_1:0conv2d_2/kernel/Adam_1/Assignconv2d_2/kernel/Adam_1/read:0
L
conv2d_2/bias/Adam:0conv2d_2/bias/Adam/Assignconv2d_2/bias/Adam/read:0
R
conv2d_2/bias/Adam_1:0conv2d_2/bias/Adam_1/Assignconv2d_2/bias/Adam_1/read:0
O
dense_1/kernel/Adam:0dense_1/kernel/Adam/Assigndense_1/kernel/Adam/read:0
U
dense_1/kernel/Adam_1:0dense_1/kernel/Adam_1/Assigndense_1/kernel/Adam_1/read:0
I
dense_1/bias/Adam:0dense_1/bias/Adam/Assigndense_1/bias/Adam/read:0
O
dense_1/bias/Adam_1:0dense_1/bias/Adam_1/Assigndense_1/bias/Adam_1/read:0
O
dense_2/kernel/Adam:0dense_2/kernel/Adam/Assigndense_2/kernel/Adam/read:0
U
dense_2/kernel/Adam_1:0dense_2/kernel/Adam_1/Assigndense_2/kernel/Adam_1/read:0
I
dense_2/bias/Adam:0dense_2/bias/Adam/Assigndense_2/bias/Adam/read:0
O
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0(��       ��-	<vzYc�A*

loss�W@�Hy;       ��-	��vzYc�A*

loss�@���^       ��-	]�wzYc�A*

loss6�@�<�       ��-	�PxzYc�A*

loss�	@o���       ��-	Z�xzYc�A*

loss^@'g       ��-	6zzYc�A*

lossSe@C�JJ       ��-	6�zzYc�A*

loss�@<�       ��-	){{zYc�A*

loss9�@�@       ��-	�&|zYc�A	*

loss�@��       ��-	S�|zYc�A
*

loss���?��       ��-	�s}zYc�A*

loss\:�?z���       ��-	;~zYc�A*

loss$��?m#9O       ��-	^�~zYc�A*

lossJ��?���
       ��-	*�zYc�A*

loss�`�? �R       ��-	�.�zYc�A*

loss��?�P�       ��-	�ـzYc�A*

loss�3�?
A�s       ��-	!u�zYc�A*

loss]I�?0k|:       ��-	��zYc�A*

loss}��?��	3       ��-	��zYc�A*

loss�8�?˱=�       ��-	B[�zYc�A*

loss?��?堀R       ��-	��zYc�A*

loss��w?����       ��-	r��zYc�A*

loss��y?���       ��-	c��zYc�A*

loss���?=?��       ��-	$C�zYc�A*

loss1:�?�M�       ��-	#܆zYc�A*

lossf�t?��'2       ��-	"q�zYc�A*

lossC=o?�q	       ��-	t
�zYc�A*

loss,�c?���f       ��-	���zYc�A*

loss6�?�:        ��-	�5�zYc�A*

loss�yk?��       ��-	�ȉzYc�A*

loss�Z?�gK0       ��-	�`�zYc�A*

loss�K?B���       ��-	 �zYc�A *

loss�Xm?g�j�       ��-	/��zYc�A!*

loss��?���F       ��-	L6�zYc�A"*

lossj.?qX/4       ��-	!ʌzYc�A#*

losshg?�{to       ��-	�]�zYc�A$*

loss�P?N�;9       ��-	A��zYc�A%*

loss�t?A��o       ��-	e��zYc�A&*

loss��?:E��       ��-	�"�zYc�A'*

loss�h�?�WH9       ��-	ຏzYc�A(*

loss�=B?�&�#       ��-	/R�zYc�A)*

lossl�?I���       ��-	}�zYc�A**

loss��!?Q4{8       ��-	��zYc�A+*

loss��5?,�]4       ��-	j�zYc�A,*

lossI�6?��k       ��-	Ǽ�zYc�A-*

lossƩ!?6��       ��-	 R�zYc�A.*

loss4-?��N�       ��-	� �zYc�A/*

loss���>N       ��-	���zYc�A0*

lossH?? ��t       ��-	�E�zYc�A1*

loss��
?,�c�       ��-	�K�zYc�A2*

loss��?^S)       ��-	�zYc�A3*

loss�O?4���       ��-	�D�zYc�A4*

loss�?����       ��-	m�zYc�A5*

loss.Մ? �       ��-	肙zYc�A6*

lossM��>��2*       ��-	��zYc�A7*

loss)��>��WZ       ��-	�v�zYc�A8*

loss
��>���J       ��-	��zYc�A9*

loss�>}ܠB       ��-	_��zYc�A:*

loss�`N?���       ��-	"U�zYc�A;*

lossm.�>_���       ��-	w��zYc�A<*

loss��?y�#       ��-	���zYc�A=*

loss_0�>R��       ��-	I�zYc�A>*

loss�H?~��       ��-	��zYc�A?*

loss(K?����       ��-	���zYc�A@*

lossL%?�H��       ��-	2�zYc�AA*

loss,�?��V�       ��-	���zYc�AB*

loss�bS?���^       ��-	�V�zYc�AC*

lossG?$�S�       ��-	��zYc�AD*

lossnX?g�       ��-	��zYc�AE*

lossh?�ŵ       ��-	��zYc�AF*

loss��?�,?�       ��-	갤zYc�AG*

loss�?�,4�       ��-	AG�zYc�AH*

lossD^�>���       ��-	�ݥzYc�AI*

loss<q
?�6Z       ��-	�u�zYc�AJ*

loss|2?���f       ��-	�zYc�AK*

loss��/?1o��       ��-	Ψ�zYc�AL*

loss��V??P!       ��-	�>�zYc�AM*

loss�/_?�$       ��-	+٨zYc�AN*

loss�>�Հ       ��-	zo�zYc�AO*

loss#��>)�/       ��-	��zYc�AP*

loss�P?0�       ��-	ß�zYc�AQ*

loss�/S?`��H       ��-	�<�zYc�AR*

loss#-?��q       ��-	�ҫzYc�AS*

loss��>��$�       ��-	�f�zYc�AT*

losst��>�_�       ��-	S%�zYc�AU*

lossW�>4�ҟ       ��-	��zYc�AV*

loss(�>�uD       ��-	��zYc�AW*

loss��	?.0�       ��-	���zYc�AX*

lossL?�N       ��-	YQ�zYc�AY*

loss��?S�:�       ��-	���zYc�AZ*

lossJ?�6��       ��-	���zYc�A[*

losss��>�>�       ��-	�/�zYc�A\*

loss;�?�87%       ��-	�ŲzYc�A]*

loss�?�2�       ��-	h�zYc�A^*

loss�?�$�       ��-	��zYc�A_*

loss�D�>=�m�       ��-	]��zYc�A`*

loss�y�>Oo�       ��-	�@�zYc�Aa*

loss%�A?���       ��-	u�zYc�Ab*

loss!�>?Ǭ�       ��-	C��zYc�Ac*

loss�
?����       ��-	�P�zYc�Ad*

loss8�>�c:       ��-	�$�zYc�Ae*

loss,8�>ї�       ��-	��zYc�Af*

loss�U?,V��       ��-	�6�zYc�Ag*

loss���>m��       ��-	�V�zYc�Ah*

loss���>4!D       ��-	�
�zYc�Ai*

loss�`�>ݣ�       ��-	CżzYc�Aj*

loss�L�>���=       ��-	�j�zYc�Ak*

loss���>n�{       ��-	q��zYc�Al*

lossK/?�׬       ��-	U��zYc�Am*

loss��?�P3       ��-	,��zYc�An*

loss�u?��3�       ��-	���zYc�Ao*

loss�6?��$       ��-	��zYc�Ap*

lossH׾>J��       ��-	���zYc�Aq*

lossĦ�>���3       ��-	@�zYc�Ar*

loss��>;�p       ��-	���zYc�As*

loss���>GO       ��-	��zYc�At*

loss�o�>,�       ��-	"�zYc�Au*

loss;�>�	W�       ��-	<��zYc�Av*

loss�/�>�E��       ��-	�Z�zYc�Aw*

lossM��>C���       ��-	)>�zYc�Ax*

loss.!�>��Ď       ��-	���zYc�Ay*

loss5�>����       ��-	ڏ�zYc�Az*

loss�s�>����       ��-	�3�zYc�A{*

lossl��>\$J       ��-	n0�zYc�A|*

loss��9>.֬       ��-	u��zYc�A}*

loss���>j���       ��-	�7�zYc�A~*

lossY~?jCŔ       ��-	��zYc�A*

loss���>�'       �	���zYc�A�*

loss���>1�m       �	j��zYc�A�*

loss�x�>���4       �	_��zYc�A�*

loss��>]]�       �	�{�zYc�A�*

lossMU�>� �$       �	y;�zYc�A�*

lossLJO>"2�       �	���zYc�A�*

loss�;|>�� >       �	��zYc�A�*

loss.E�>pM �       �	!;�zYc�A�*

lossѺ�>6prS       �	���zYc�A�*

loss�<�>���       �	>x�zYc�A�*

loss�n�>$�OR       �	�#�zYc�A�*

loss#�>��8       �	���zYc�A�*

loss�3X>� p       �	=b�zYc�A�*

loss�->��N$       �		�zYc�A�*

loss�ι>���I       �	��zYc�A�*

lossW�>��       �	ZK�zYc�A�*

loss���>n��       �	��zYc�A�*

loss��>UK��       �	��zYc�A�*

loss	�[>�N�z       �	�'�zYc�A�*

lossxq�>dn��       �	���zYc�A�*

loss�((>}@��       �	�g�zYc�A�*

lossWBs>��I       �		�zYc�A�*

loss��>��b       �	E��zYc�A�*

losslOt>o�f�       �	�8�zYc�A�*

loss}��>�A       �	���zYc�A�*

lossű�>����       �	ǂ�zYc�A�*

loss�ҭ>"<^       �	�"�zYc�A�*

lossft}>`u�P       �	���zYc�A�*

lossn�%>��_       �	�]�zYc�A�*

loss�1�>d��X       �	��zYc�A�*

lossN�>���       �	1��zYc�A�*

loss��>����       �	�a�zYc�A�*

loss��>nd8�       �	��zYc�A�*

lossS�>
}O       �	���zYc�A�*

loss���>1<��       �	rk�zYc�A�*

loss���>_�bW       �	��zYc�A�*

loss��>6�-U       �	���zYc�A�*

lossvq�>��+       �	�U�zYc�A�*

loss�Z>���       �	���zYc�A�*

loss�>*A=       �	��zYc�A�*

loss#m�>Y^       �	�>�zYc�A�*

loss0J>�p$       �	���zYc�A�*

loss��a>4c��       �	���zYc�A�*

loss~ۋ>�       �	���zYc�A�*

loss��6>���-       �	XW�zYc�A�*

lossR8�>�O��       �	&��zYc�A�*

loss$�>�t�       �	I��zYc�A�*

loss�t>bH�       �	VI�zYc�A�*

loss7�>���       �	���zYc�A�*

lossw�?�!pG       �	7��zYc�A�*

loss���>D�}       �	�,�zYc�A�*

loss,T�>Ɖ˥       �	`��zYc�A�*

loss��>y���       �	fi�zYc�A�*

loss۱=�<?       �	=�zYc�A�*

lossZj�>����       �	<��zYc�A�*

lossV7�>{�{�       �	VE�zYc�A�*

lossM>Ŏ>       �	�@�zYc�A�*

loss��w>��       �	�G�zYc�A�*

lossxz�>��fO       �	�y�zYc�A�*

lossN�t>�2��       �	�K�zYc�A�*

loss�9�>���       �	���zYc�A�*

loss�+�>2���       �	��zYc�A�*

loss�E�>m�ӌ       �	4��zYc�A�*

lossL$�>���y       �	Y��zYc�A�*

loss��>ilzV       �	��zYc�A�*

loss��z>�S       �	���zYc�A�*

lossD��>���       �	{�zYc�A�*

loss!�Q>���       �	�?�zYc�A�*

lossZ=i>��       �	���zYc�A�*

loss��>n�e�       �	�� {Yc�A�*

loss81�>��*�       �	��{Yc�A�*

loss���>���       �	�6{Yc�A�*

lossia�>��k�       �	{Yc�A�*

lossH��>��Ҵ       �	�B{Yc�A�*

loss�i]>$���       �	E�{Yc�A�*

lossd�z>*���       �	ao{Yc�A�*

loss���><        �	"P{Yc�A�*

loss e_>+�z�       �	o�{Yc�A�*

lossɃ>*�13       �	N�{Yc�A�*

loss�L}>��E       �	�;	{Yc�A�*

lossL�O>v�̋       �	��	{Yc�A�*

loss�T>�b��       �	��
{Yc�A�*

lossbW�> �h)       �	l!{Yc�A�*

loss�>#�r�       �	��{Yc�A�*

loss���>D-F       �	OY{Yc�A�*

loss�>>�ʊg       �	��{Yc�A�*

loss �]>I��       �	 �{Yc�A�*

loss��>��@       �	�P{Yc�A�*

loss!�>�5~B       �	��{Yc�A�*

loss�>5�3E       �	n�{Yc�A�*

loss�n�>5g:       �	'{Yc�A�*

loss��>�uzK       �	��{Yc�A�*

loss�@�>xʠ�       �	_{Yc�A�*

loss3�)>�P��       �	f�{Yc�A�*

loss�/q>!�cB       �	T�{Yc�A�*

loss���>��       �	�%{Yc�A�*

lossx�>��q       �	#�{Yc�A�*

loss��k>��       �	OW{Yc�A�*

loss��a>��VO       �	��{Yc�A�*

loss*�>ޙ�       �	M�{Yc�A�*

loss�v�>.�3�       �	UO{Yc�A�*

loss��>�Cu�       �	��{Yc�A�*

loss-Č>Lm�       �	v{Yc�A�*

lossm�2>��R       �	{Yc�A�*

loss��/>|]E       �	m�{Yc�A�*

loss��>p���       �	3R{Yc�A�*

loss��$><vW�       �	>�{Yc�A�*

loss2�>y���       �	=�{Yc�A�*

loss@��>�o,�       �	z{Yc�A�*

loss�cI>�Ug�       �	^�{Yc�A�*

lossE�>�a�       �	Ec{Yc�A�*

loss��>��}�       �	a�{Yc�A�*

loss��>>��"�       �	J�{Yc�A�*

loss���>�{ư       �	z7{Yc�A�*

lossiE�>�9^j       �	��{Yc�A�*

loss��>��9       �	Eb{Yc�A�*

loss��Q>�>��       �	��{Yc�A�*

loss* �>۷�       �	�� {Yc�A�*

loss>]>�+/�       �	�0!{Yc�A�*

loss!�h>����       �	g�!{Yc�A�*

loss��>>�l�K       �	Kr"{Yc�A�*

lossM��>�b       �	�!#{Yc�A�*

lossLb�>�<�Z       �	#�#{Yc�A�*

loss�{�>*`       �	�Y${Yc�A�*

loss�>O��       �	!%{Yc�A�*

loss�^�>���&       �	��%{Yc�A�*

loss�>>�@�       �	hB&{Yc�A�*

lossI>�`��       �	q�&{Yc�A�*

loss-�>нW6       �	�'{Yc�A�*

lossqҜ>55��       �	�'({Yc�A�*

loss��>��$       �	��({Yc�A�*

loss��a>��       �	+l){Yc�A�*

losst׎>���p       �	�*{Yc�A�*

loss1�>��/�       �	#�*{Yc�A�*

loss��4>}�u�       �	�[+{Yc�A�*

loss%��>[���       �	��+{Yc�A�*

loss#hM>]ذ�       �	2�,{Yc�A�*

loss�i�>���       �	�&-{Yc�A�*

lossV
v>Vz	       �	�.{Yc�A�*

loss��>b�v�       �	�.{Yc�A�*

loss�Θ>R�ұ       �	m6/{Yc�A�*

loss=��>�n       �	>�/{Yc�A�*

loss�p>�`,�       �	�`0{Yc�A�*

loss�y�>)��       �	<�0{Yc�A�*

loss��r>��a       �	r�1{Yc�A�*

loss��u>���O       �	}!2{Yc�A�*

loss�Sf>��Η       �	o�2{Yc�A�*

loss'�>�P5       �	�Z3{Yc�A�*

loss�f�>���?       �	%4{Yc�A�*

loss	>@�       �	5�4{Yc�A�*

lossVR>���       �	,5{Yc�A�*

loss���>�>�       �	��5{Yc�A�*

lossW�U>V�xg       �	*V6{Yc�A�*

loss�m>�k��       �	#f7{Yc�A�*

loss�[)>pH       �	08{Yc�A�*

lossi�L>{��y       �	>�8{Yc�A�*

loss��%>���W       �	�?9{Yc�A�*

loss:��>�        �	�9{Yc�A�*

loss�ۈ>�V�x       �	p:{Yc�A�*

loss�*>���       �	;{Yc�A�*

lossEC�=�lm�       �	�;{Yc�A�*

losss s>z煠       �	]3<{Yc�A�*

loss�>�=�:>       �	��<{Yc�A�*

lossW+�=�ȱ        �	�`={Yc�A�*

loss���>�S�       �	Q�={Yc�A�*

loss3n>���       �	*�>{Yc�A�*

loss��=�Ⱥ       �	&?{Yc�A�*

lossZ'>XDLj       �	;�?{Yc�A�*

loss�=d>��c�       �	Ze@{Yc�A�*

lossW#Z>f�Z�       �	�A{Yc�A�*

loss��O>�E�       �	I�A{Yc�A�*

loss�,�>4ǋ�       �	�bC{Yc�A�*

loss�g>DzA       �	M�C{Yc�A�*

losslU>i�!       �	N�D{Yc�A�*

lossMj>yjwI       �	�>E{Yc�A�*

lossA:]>_[��       �	��E{Yc�A�*

loss�J>���       �	PsF{Yc�A�*

loss��>(�       �	�G{Yc�A�*

loss;��>�7k?       �	�H{Yc�A�*

loss)KT=����       �	�H{Yc�A�*

lossݸ�>��}       �	�4I{Yc�A�*

loss��=O\z[       �	��I{Yc�A�*

loss{ =>H�^�       �	W{J{Yc�A�*

loss��I>�y��       �	�K{Yc�A�*

loss��>f@=3       �	�K{Yc�A�*

loss�\S>t��       �	��L{Yc�A�*

lossT�=?���       �	9~M{Yc�A�*

loss�$5>5�n       �	2N{Yc�A�*

lossH�>z�       �	4�N{Yc�A�*

lossj�V>�ً�       �	�dO{Yc�A�*

losscn >L���       �	�P{Yc�A�*

lossQ�>�
Z�       �	L�P{Yc�A�*

loss�m>�F��       �	vSR{Yc�A�*

loss��>{�       �	��R{Yc�A�*

lossZ�>�I7       �	��S{Yc�A�*

loss��>�Gơ       �	()T{Yc�A�*

loss\R>�E!�       �	��T{Yc�A�*

loss��>��3       �	�mU{Yc�A�*

lossL�?�}A       �	ȔV{Yc�A�*

losso��>�?X�       �	�8W{Yc�A�*

loss��I>}؀       �	��W{Yc�A�*

loss�݀>�&G       �	�yX{Yc�A�*

loss�+�>�       �	YY{Yc�A�*

loss�1[>�첟       �	��Y{Yc�A�*

lossM�>"]��       �	�Z{Yc�A�*

loss��>�       �	/i[{Yc�A�*

loss�u>~t%�       �	X\{Yc�A�*

loss�!�>yu@n       �	�\{Yc�A�*

loss�o�=�ޥx       �	XV]{Yc�A�*

lossH�h>�{Q8       �	v�]{Yc�A�*

lossAޠ>z��       �	�^{Yc�A�*

loss,�p>��{4       �	�@_{Yc�A�*

loss�o�=���       �	��_{Yc�A�*

loss�G2>oG�F       �	�`{Yc�A�*

loss�7>4�)�       �	�:a{Yc�A�*

loss]�t>Yq��       �	��a{Yc�A�*

loss��'>�:�       �	A�b{Yc�A�*

loss�b?�r�       �	�Ec{Yc�A�*

loss.*>�*       �	:�c{Yc�A�*

loss$? >��s[       �	�d{Yc�A�*

lossѨ]>�O�       �	�9e{Yc�A�*

loss}�D>)�y�       �	7�e{Yc�A�*

loss�B�>�kC       �	�f{Yc�A�*

loss�V?>�G H       �	#g{Yc�A�*

loss%��=د�_       �	��g{Yc�A�*

loss(e�==�Wa       �	�ih{Yc�A�*

loss%>z��       �	i{Yc�A�*

loss!�V>��E       �	��i{Yc�A�*

loss_~>�-       �	Mij{Yc�A�*

loss��>�b�       �	�k{Yc�A�*

loss�C>	Y�       �	�k{Yc�A�*

lossc�>P),       �	�Xl{Yc�A�*

loss�>��<X       �	�m{Yc�A�*

loss5��>�A@�       �	ɫm{Yc�A�*

lossg>m��       �	/Rn{Yc�A�*

loss�.>|��       �	yo{Yc�A�*

lossi�>E��       �	&�o{Yc�A�*

lossܷ�>BK�7       �	HQp{Yc�A�*

loss��>B5��       �	�p{Yc�A�*

lossہG>��       �	��q{Yc�A�*

lossԮ�=�d�[       �	3Sr{Yc�A�*

loss�>��U�       �	��r{Yc�A�*

losso|!>���       �	��s{Yc�A�*

loss��>�#��       �	Mt{Yc�A�*

loss�~�=;Mh       �	��t{Yc�A�*

lossL4>�ag       �	�u{Yc�A�*

loss�O>����       �	�Cv{Yc�A�*

lossMRM>����       �	��v{Yc�A�*

loss	
�>Io��       �	��w{Yc�A�*

loss3�]>��       �	�%x{Yc�A�*

loss@!,>V�)       �	��x{Yc�A�*

losss8,>� N!       �	��y{Yc�A�*

loss��G>�o��       �	Rz{Yc�A�*

loss*��>ϫI       �	��z{Yc�A�*

loss��w>��AM       �	��{{Yc�A�*

lossMyL>3�bT       �	\r|{Yc�A�*

loss�E�=��C�       �	37}{Yc�A�*

loss:	>*<�       �	�}{Yc�A�*

lossoX >n�       �	̘~{Yc�A�*

loss�'>�Xa       �	q<{Yc�A�*

loss�:>�ǣi       �	%�{Yc�A�*

loss/�>�J�       �	N��{Yc�A�*

loss�L.>���       �	�^�{Yc�A�*

lossqD>���,       �	 q�{Yc�A�*

loss@��=�9�u       �	�	�{Yc�A�*

lossݯz>��{�       �	���{Yc�A�*

lossZhp>�"�       �	ߋ�{Yc�A�*

lossf�>R�       �	�4�{Yc�A�*

loss�w>��T       �	���{Yc�A�*

loss�"�=��{       �	w�{Yc�A�*

loss[>��:�       �	��{Yc�A�*

loss��2>��f       �	�{Yc�A�*

loss�{I>=���       �	�g�{Yc�A�*

loss�k>�nx"       �	�n�{Yc�A�*

loss��X>�?�       �	��{Yc�A�*

loss�0>Ƽ:i       �	7��{Yc�A�*

loss�?�=��)       �	���{Yc�A�*

lossd��=���       �	v6�{Yc�A�*

loss��t>�"�j       �	?�{Yc�A�*

loss�D�>�1�       �	�ʍ{Yc�A�*

lossLTv>5�K       �	2��{Yc�A�*

loss�&>H��x       �	 ď{Yc�A�*

losso@�>���       �	�f�{Yc�A�*

loss�4>(��;       �	J)�{Yc�A�*

loss�2B>J�y       �	Ց{Yc�A�*

lossO�>�?�       �	it�{Yc�A�*

lossi��>^��       �	uZ�{Yc�A�*

loss��>Sy�_       �	�{Yc�A�*

lossI(�>��Z       �	���{Yc�A�*

lossL�~=���       �	Yi�{Yc�A�*

loss��>�C��       �	���{Yc�A�*

loss��W>Z�       �	���{Yc�A�*

loss=�)>�Ԍ       �	�×{Yc�A�*

loss�u>�h�!       �	Ѳ�{Yc�A�*

loss6��=d�O       �	cc�{Yc�A�*

loss6h�=$�u-       �	�6�{Yc�A�*

loss�҇>��(       �	���{Yc�A�*

loss�b>aȻ�       �	���{Yc�A�*

loss�$�>����       �	_��{Yc�A�*

loss�">�G/�       �	t^�{Yc�A�*

loss�n�>��wF       �	+��{Yc�A�*

loss��>��e�       �	�ߞ{Yc�A�*

loss��=��Vn       �	d̟{Yc�A�*

loss1��>����       �	��{Yc�A�*

lossZ�J>LUz       �	�8�{Yc�A�*

loss4�=���l       �	�ӡ{Yc�A�*

loss]�>�0��       �	^��{Yc�A�*

lossN+�>l�z�       �	�{Yc�A�*

loss���=�H.       �	��{Yc�A�*

loss��=��Um       �	Ϣ�{Yc�A�*

loss'j>9�M�       �	[?�{Yc�A�*

loss |�=����       �	��{Yc�A�*

lossB�=�u       �	'��{Yc�A�*

loss��>|)�       �	�$�{Yc�A�*

lossw�>��.)       �	��{Yc�A�*

loss >��e       �	c�{Yc�A�*

loss��/>�0�       �	6�{Yc�A�*

loss���=�츶       �	M��{Yc�A�*

lossc�E>I��       �	|c�{Yc�A�*

loss�D?�%J�       �	���{Yc�A�*

loss;��>��<       �	R��{Yc�A�*

lossd?�='��        �		3�{Yc�A�*

loss�>�̨{       �	6ˬ{Yc�A�*

loss]O>om%       �	�a�{Yc�A�*

lossx��= W��       �	r��{Yc�A�*

lossO^�=�>��       �	ɒ�{Yc�A�*

lossN�W>N�4       �	=(�{Yc�A�*

loss��>���       �	M��{Yc�A�*

loss�w>#6��       �	Ie�{Yc�A�*

loss~b�>='�E       �	���{Yc�A�*

loss�M>!� �       �	7߱{Yc�A�*

loss�[M>f���       �	�v�{Yc�A�*

loss8�#>"S|       �	�{Yc�A�*

loss���=(���       �	䣳{Yc�A�*

loss�=5��       �	C8�{Yc�A�*

loss��=_�3�       �	Ҵ{Yc�A�*

loss�7b>��       �	�j�{Yc�A�*

loss��=�N+;       �	=
�{Yc�A�*

loss�k>ܗ^       �	���{Yc�A�*

loss�p�=R���       �	D�{Yc�A�*

loss���=����       �	��{Yc�A�*

loss*z�=�(�L       �	#,�{Yc�A�*

lossu��=x��       �	HĹ{Yc�A�*

loss�,>S|��       �	x�{Yc�A�*

loss��%>Ȱ�r       �	k'�{Yc�A�*

loss��[>����       �	�ݻ{Yc�A�*

loss��>M+��       �	��{Yc�A�*

loss�>,��       �	�`�{Yc�A�*

loss���=��]�       �	Q��{Yc�A�*

loss�:�=�i��       �	�G�{Yc�A�*

lossX%>�Ԃo       �	�<�{Yc�A�*

loss| I>�g��       �	K��{Yc�A�*

loss)O�>[úB       �	9��{Yc�A�*

loss��>E8;�       �	�L�{Yc�A�*

loss�=>Y�       �	���{Yc�A�*

lossc�>T��       �	r��{Yc�A�*

lossÄ?>���       �	!�{Yc�A�*

loss��>���       �	���{Yc�A�*

loss��=U�.�       �	�m�{Yc�A�*

loss*��>!�]       �	��{Yc�A�*

lossʲ>>_��4       �	e��{Yc�A�*

loss�L�=����       �	�D�{Yc�A�*

losst`|>C&�       �	X��{Yc�A�*

lossY=>sߜ_       �	˂�{Yc�A�*

loss��>>n�K�       �	b��{Yc�A�*

lossV�=����       �	t��{Yc�A�*

loss؁=���d       �	b0�{Yc�A�*

loss-N>�v�Z       �	�{Yc�A�*

loss�I	>i��P       �	/��{Yc�A�*

loss��>f��r       �	���{Yc�A�*

loss�=y� �       �	�^�{Yc�A�*

loss!�=|��       �	v��{Yc�A�*

loss�>]=       �	K��{Yc�A�*

loss�z>�&��       �	l&�{Yc�A�*

losse�=����       �	r��{Yc�A�*

loss�]�=_�       �	=�{Yc�A�*

loss�-�=�(��       �	L�{Yc�A�*

loss��;>�;�8       �	Ϻ�{Yc�A�*

loss���=@&       �	+k�{Yc�A�*

lossQ�>KԀ�       �	d�{Yc�A�*

loss�P�=Z��       �	��{Yc�A�*

loss1�>T, >       �	�@�{Yc�A�*

lossT9>r���       �	���{Yc�A�*

loss�}A=�"��       �	���{Yc�A�*

loss�P>���       �	2�{Yc�A�*

lossb�=�߹�       �	���{Yc�A�*

loss��g>���       �	k��{Yc�A�*

loss>�>$Q{       �	,��{Yc�A�*

loss�,>�|]^       �	MH�{Yc�A�*

lossϑ�=`�{�       �	���{Yc�A�*

lossܯ1>��g       �	r��{Yc�A�*

loss$��=���       �	�c�{Yc�A�*

loss�%>!׵�       �	���{Yc�A�*

lossF�=Ij�a       �	=��{Yc�A�*

lossi�$>Q��       �	f��{Yc�A�*

loss��h=("�4       �	y"�{Yc�A�*

loss�>�F�n       �	��{Yc�A�*

loss��=07&       �	���{Yc�A�*

loss�d�=�w8       �	2�{Yc�A�*

loss	R>�8�       �	���{Yc�A�*

lossv�+>7z�       �	f��{Yc�A�*

loss�Eg>Q���       �	�9�{Yc�A�*

loss��U=��-       �	���{Yc�A�*

lossOgj=�x{�       �	�u�{Yc�A�*

lossɃ�=����       �	��{Yc�A�*

loss`�C=w͟       �	Ҧ�{Yc�A�*

loss=)1=Wz��       �	yw�{Yc�A�*

loss�N=6K�       �	� �{Yc�A�*

loss�>��       �	���{Yc�A�*

loss��T=��+       �	/R�{Yc�A�*

loss
!v=C���       �	���{Yc�A�*

loss�-=�$	�       �	�~�{Yc�A�*

loss�l>I5H       �	�!�{Yc�A�*

loss4�z<R�       �	��{Yc�A�*

loss�|<���       �	�Q�{Yc�A�*

loss*p^<�a��       �	,��{Yc�A�*

loss��>�ru       �	��{Yc�A�*

lossۗJ>�Ɨ�       �	/�{Yc�A�*

lossm�>��       �	l��{Yc�A�*

loss8^�<��!       �	�e�{Yc�A�*

loss#J>6��       �	P��{Yc�A�*

loss��?����       �	n��{Yc�A�*

lossog�<V�       �	�A�{Yc�A�*

lossA�^>5�V       �	.��{Yc�A�*

lossh��=��p6       �	��{Yc�A�*

loss�$�>eNH�       �	q!�{Yc�A�*

lossAA�=�$j3       �	���{Yc�A�*

loss QR=�:�       �	kb�{Yc�A�*

loss�UX>F�M       �	� �{Yc�A�*

loss!�->�}�       �	k��{Yc�A�*

lossT�l>��p       �	�:�{Yc�A�*

loss&l6>6^H�       �	���{Yc�A�*

loss��)>��p       �	c{�{Yc�A�*

lossM��>�_rA       �	�{Yc�A�*

loss�V>�~�       �	~V�{Yc�A�*

loss
.>m��-       �	v��{Yc�A�*

lossV�>�%y�       �	=��{Yc�A�*

loss�n�>�o�       �	*8�{Yc�A�*

loss�0�=l<�8       �	��{Yc�A�*

loss�0�='�r       �	�s�{Yc�A�*

loss��>�GE�       �	��{Yc�A�*

loss��>�&{       �	���{Yc�A�*

loss�)�=>,6       �	�O�{Yc�A�*

loss/X>�ڼ,       �	���{Yc�A�*

loss�H�=��2+       �	8��{Yc�A�*

loss�0g=8V�.       �	=F�{Yc�A�*

loss��=C�j@       �	���{Yc�A�*

loss]�=!!�       �	�� |Yc�A�*

loss.#Q>Ƴ�a       �	1D|Yc�A�*

loss$��=�b~�       �	��|Yc�A�*

lossD1>�J�Q       �	�|Yc�A�*

lossP�!>���       �	A*|Yc�A�*

loss���=��@       �	.�|Yc�A�*

loss�m�=ѺE^       �	 _|Yc�A�*

loss�>�wL1       �	��|Yc�A�*

loss�C�=�l�       �	l�|Yc�A�*

loss8�=\��       �	�,|Yc�A�*

loss?��=�Q       �	��|Yc�A�*

loss�͍=����       �	�[|Yc�A�*

lossW�/>�\�9       �	��|Yc�A�*

loss�~:>�u�       �	��|Yc�A�*

lossf�;>�ɐ�       �	�&	|Yc�A�*

lossZ�=9�<�       �	W�	|Yc�A�*

loss]�=
F��       �	�b
|Yc�A�*

lossz�+>}%d       �	��
|Yc�A�*

lossW�>Y4��       �	y�|Yc�A�*

loss<߿=�\d       �	Eh|Yc�A�*

loss�� >���,       �	�F|Yc�A�*

lossJ; >����       �	��|Yc�A�*

loss�3>jg�       �	Sy|Yc�A�*

loss�F>I�3�       �	f|Yc�A�*

loss ��=|$       �	'�|Yc�A�*

loss�_�=d�[N       �	dZ|Yc�A�*

loss��A>)і       �	�
*|Yc�A�*

loss�1�=��d�       �	ƣ*|Yc�A�*

loss��t>tL7G       �	0G+|Yc�A�*

lossr��=�t<       �	�+|Yc�A�*

loss�h	>K��       �	�t,|Yc�A�*

loss\/�=c�M�       �	�-|Yc�A�*

loss ��=�-       �	C�-|Yc�A�*

loss�d�=���       �	A.|Yc�A�*

lossF�@>Ii2       �	��.|Yc�A�*

loss�T>˔'"       �	��/|Yc�A�*

loss��=�%e       �	^M0|Yc�A�*

lossэ=���       �	?�0|Yc�A�*

loss�y�=�[e�       �	5�1|Yc�A�*

loss�M�=�_��       �	bj2|Yc�A�*

loss��=�8l       �	3|Yc�A�*

loss�9>U5w�       �	D�3|Yc�A�*

loss/�u="��5       �	�84|Yc�A�*

loss�{�=�n{�       �	E�4|Yc�A�*

loss�(�=�?��       �	8k5|Yc�A�*

loss@��>�Z]v       �	m�5|Yc�A�*

loss���=o�H�       �	�6|Yc�A�*

loss��d>\x��       �	�*7|Yc�A�*

loss,i�=޹�       �	�7|Yc�A�*

loss�0T>����       �	�T8|Yc�A�*

loss�h)>]��       �	9|Yc�A�*

loss[�=��u�       �	�B:|Yc�A�*

lossi�=���       �	��:|Yc�A�*

loss��>Ǌ       �	�;|Yc�A�*

loss��!>���       �	��<|Yc�A�*

loss��k>�_        �	�z=|Yc�A�*

loss�>T.�L       �	�/>|Yc�A�*

loss3��=���3       �	��>|Yc�A�*

loss)�E>��       �	��?|Yc�A�*

lossx�v>��-�       �	Y@|Yc�A�*

loss
��=�g       �	�*A|Yc�A�*

loss��>��[       �	��A|Yc�A�*

loss�v=*c��       �	��B|Yc�A�*

loss�>9�q       �	�xC|Yc�A�*

loss�V�>=��+       �	/nD|Yc�A�*

loss�F�>}�in       �	�E|Yc�A�*

lossM>㴨�       �	v�E|Yc�A�*

loss���=��       �	�~F|Yc�A�*

lossdq�=d�       �	�G|Yc�A�*

loss��^>Q���       �	��G|Yc�A�*

loss�R$>O�)       �	�_H|Yc�A�*

loss,m3>�0��       �	��H|Yc�A�*

loss�N>V�[       �	&�I|Yc�A�*

loss� �=�2W       �	�"J|Yc�A�*

loss��=v	�       �	Z�J|Yc�A�*

loss��=S*�       �	�HK|Yc�A�*

loss;D�=�u��       �	��K|Yc�A�*

lossø4>}�7�       �	rpL|Yc�A�*

lossMY'>�]�Q       �	cM|Yc�A�*

loss�M?E�(       �	?�M|Yc�A�*

loss�9�=���N       �	sfN|Yc�A�*

loss�e=v�9�       �	��N|Yc�A�*

lossa�=g���       �	�O|Yc�A�*

loss���=#���       �	I-P|Yc�A�*

loss_�>��	       �	i�P|Yc�A�*

loss�w�=Jw��       �	F\Q|Yc�A�*

loss��#>E��       �	.�Q|Yc�A�*

loss���=��r�       �	��R|Yc�A�*

loss @�=r5        �	>S|Yc�A�*

lossܾ3>����       �	0�S|Yc�A�*

loss��=�#��       �	mT|Yc�A�*

losso�=��u       �	�U|Yc�A�*

loss:�+>�yv       �	�U|Yc�A�*

loss;�@>�u�       �	�-V|Yc�A�*

loss,M�>ҳ       �	i�V|Yc�A�*

loss#Y>+>r�       �	UhW|Yc�A�*

lossr��=Ļ�X       �	zRX|Yc�A�*

loss��=y(�       �	b�X|Yc�A�*

loss)�R=��-�       �	l�Y|Yc�A�*

loss�*>�Z��       �	�HZ|Yc�A�*

lossZ�r>{�ջ       �	��Z|Yc�A�*

losso�>@���       �	�{[|Yc�A�*

lossL�g>��q�       �	+\|Yc�A�*

loss�7i>�?�M       �	�\|Yc�A�*

lossZX�=X��n       �	~�]|Yc�A�*

loss:�=*�{       �	�^|Yc�A�*

loss�s>��P       �	Ui_|Yc�A�*

lossM[>��`z       �	m `|Yc�A�*

loss�]G>��-@       �	��`|Yc�A�*

lossx}=}=7�       �	�aa|Yc�A�*

loss %1>����       �	�b|Yc�A�*

lossD]�=Yws       �	c|Yc�A�*

loss3�l=c�O�       �	Ϊc|Yc�A�*

lossh>�^s�       �	}@d|Yc�A�*

loss}u�=��mA       �	Y�d|Yc�A�*

loss���=K�lz       �	�se|Yc�A�*

loss(�c>�&i�       �	�f|Yc�A�*

lossߋ->,(I�       �	��f|Yc�A�*

loss�$>���       �	1g|Yc�A�*

loss!�'>y��       �	�	h|Yc�A�*

loss��>U��       �	�h|Yc�A�*

lossa.�=�}n       �	�5i|Yc�A�*

loss�=&>ʢ��       �	0�i|Yc�A�*

loss��E>T���       �	Ulj|Yc�A�*

lossʦ�>B��       �	�k|Yc�A�*

loss���= a�_       �	��k|Yc�A�*

loss��=�*��       �	;Tl|Yc�A�*

loss� v=#�@�       �	�5m|Yc�A�*

loss��*>�[:�       �	p�m|Yc�A�*

loss�/�=�Rg�       �	`rn|Yc�A�*

loss�]�=���v       �	Wo|Yc�A�*

loss$ut>_���       �	V�o|Yc�A�*

loss=PJx       �	/p|Yc�A�*

loss,�{=L��       �	y�p|Yc�A�*

loss�(a>t��       �	�dq|Yc�A�*

loss��=�8Ș       �	;�q|Yc�A�*

loss��>�Z8       �	�Ls|Yc�A�*

loss���=f��=       �	��s|Yc�A�*

lossw'�=��Z�       �	 �t|Yc�A�*

loss��7=��7       �	�,u|Yc�A�*

lossT�>���       �	z�u|Yc�A�*

loss苚=�^�       �	�^v|Yc�A�*

loss�z�=1�߿       �	 w|Yc�A�*

loss�R�=^{G       �	|�w|Yc�A�*

loss��=>�`�       �	�9x|Yc�A�*

loss`��=���c       �	��x|Yc�A�*

loss5>l��       �	uty|Yc�A�*

loss�5�=U�"       �	�z|Yc�A�*

loss�P>܅)       �	֪z|Yc�A�*

loss�G>����       �	�D{|Yc�A�*

loss�f�<����       �	3�{|Yc�A�*

lossI�M>y�k       �	${||Yc�A�*

lossr�9>��       �	�}|Yc�A�*

loss@��=�د       �	�}|Yc�A�*

loss;-?>��       �	@k~|Yc�A�*

loss��=��c       �	>|Yc�A�*

loss�	H>bd��       �	ؚ|Yc�A�*

loss>�^�       �	7�|Yc�A�*

loss���=���G       �	�΀|Yc�A�*

loss8` =h�C#       �	u�|Yc�A�*

lossΘV>�M�8       �	S�|Yc�A�*

loss!��=gW�1       �	���|Yc�A�*

loss��=	�       �	�>�|Yc�A�*

loss��-=� �       �	Ԅ|Yc�A�*

loss�O= �$�       �	�m�|Yc�A�*

loss�=B�%       �	��|Yc�A�*

lossNW=�A��       �	ض�|Yc�A�*

loss���=z{�       �	vR�|Yc�A�*

loss��`>��L       �	��|Yc�A�*

lossA�>	���       �	:��|Yc�A�*

loss�6[>�M-       �	�>�|Yc�A�*

loss��"=dG�       �	�ډ|Yc�A�*

loss�P	>�6�3       �	>{�|Yc�A�*

loss�Q�=��
)       �	��|Yc�A�*

loss|��=�i�       �	Ͽ�|Yc�A�*

loss�x>ȯ��       �	�T�|Yc�A�*

lossK�=iۻ        �	��|Yc�A�*

loss��C>��؃       �	���|Yc�A�*

lossvR�=�Tօ       �	��|Yc�A�*

lossO�2>���       �	���|Yc�A�*

loss���=���       �	�G�|Yc�A�*

loss=	�<>�V       �	�ݏ|Yc�A�*

loss�<>���       �	p�|Yc�A�*

loss���='��r       �	��|Yc�A�*

loss�D>ײ�(       �	(��|Yc�A�*

lossĺ=S�I?       �	�0�|Yc�A�*

lossV�>	!oj       �	Pǒ|Yc�A�*

loss��f=��*�       �	H��|Yc�A�*

loss�E�=��z       �	�R�|Yc�A�*

loss)s�<        �	F�|Yc�A�*

lossLqA>�~C       �	k��|Yc�A�*

lossA�=eloq       �	��|Yc�A�*

loss6��=�v�       �	���|Yc�A�*

loss�-�=$�7W       �	�@�|Yc�A�*

loss��=�-�       �	�ۗ|Yc�A�*

loss>��|+       �	�v�|Yc�A�*

lossh+>���W       �	��|Yc�A�*

loss�I5=�W�3       �	���|Yc�A�*

loss�>����       �	A�|Yc�A�*

loss�:>�r       �	Qܚ|Yc�A�*

loss��i=�G�       �	��|Yc�A�*

lossꈵ=���       �	A~�|Yc�A�*

loss\1>R��       �	�H�|Yc�A�*

loss4>�?�_       �	<��|Yc�A�*

loss{S>��       �	͕�|Yc�A�*

loss2�=ԌG4       �	2�|Yc�A�*

loss�k�<l�G       �	�џ|Yc�A�*

loss�>A��
       �	Dj�|Yc�A�*

losse��=����       �	t�|Yc�A�*

losszV!=N�       �	���|Yc�A�*

loss���=�v��       �	N�|Yc�A�*

loss{��=�,,       �	�J�|Yc�A�*

losscT>n��       �	�|Yc�A�*

loss��p=0�E       �	z��|Yc�A�*

loss��=�>�       �	���|Yc�A�*

loss���=BW��       �	�'�|Yc�A�*

lossݑ;>/
�       �	̷�|Yc�A�*

loss�g�=p�Vs       �	�I�|Yc�A�*

loss�S�=��F       �	�ާ|Yc�A�*

loss�=�ͤc       �	���|Yc�A�*

loss���<�x�       �	->�|Yc�A�*

loss � >��|�       �	�Щ|Yc�A�*

loss�>�G       �	�c�|Yc�A�*

loss@�=H3c#       �	���|Yc�A�*

lossLG>{���       �	L��|Yc�A�*

loss`��=� W�       �	�*�|Yc�A�*

loss��=|O��       �	�ͬ|Yc�A�*

loss<�x=�vkT       �	�d�|Yc�A�*

loss(d>��9�       �	���|Yc�A�*

loss�4�=�+-�       �	X��|Yc�A�*

loss���=��p       �	�1�|Yc�A�*

lossC��="��       �	ɯ|Yc�A�*

loss�#>�V@       �	�f�|Yc�A�*

lossR$�=-���       �	]��|Yc�A�*

loss`�>�!�       �	G��|Yc�A�*

loss�Ő=T��       �	+�|Yc�A�*

loss��=�Z�       �	�Ų|Yc�A�*

loss�S�=p��       �	�`�|Yc�A�*

loss�)=a�7       �	e��|Yc�A�*

loss�m=L��       �	���|Yc�A�*

loss}3>L¨F       �	E)�|Yc�A�*

loss�H�=ܦ\�       �	�ĵ|Yc�A�*

loss.(�>ƔV       �	�Z�|Yc�A�*

lossS��>�^��       �	��|Yc�A�*

loss��Y>�4�       �	C��|Yc�A�*

lossX�9>
��       �	�!�|Yc�A�*

losscQ=�W��       �	$�|Yc�A�*

lossDNB=pvx�       �	~�|Yc�A�*

lossE�9>�B�       �	)v�|Yc�A�*

loss��?>�Z>Q       �	0-�|Yc�A�*

loss��<���       �	2̻|Yc�A�*

loss���=p��       �	Po�|Yc�A�*

loss��=�{P       �	F�|Yc�A�*

lossd��=]I{       �	j��|Yc�A�*

loss�|�=��$$       �	�8�|Yc�A�*

loss�߯=q*�       �	�Ӿ|Yc�A�*

loss��o=rs%       �	Dk�|Yc�A�*

loss3�<��Q        �	��|Yc�A�*

loss3�E>D�7�       �	��|Yc�A�*

lossS��=���       �	�?�|Yc�A�*

losse`/>zAQ�       �	���|Yc�A�*

loss;`�=,0�9       �	�l�|Yc�A�*

loss���=݄��       �	�|Yc�A�*

loss1!c>�*�6       �	ݳ�|Yc�A�*

loss�dW>��]�       �	�U�|Yc�A�*

loss$)=��%       �	b��|Yc�A�*

loss)�="s       �	���|Yc�A�*

loss�=,*�       �	.:�|Yc�A�*

lossWJ>��       �	���|Yc�A�*

loss���=��       �	l�|Yc�A�*

loss%�>�=X�       �	��|Yc�A�*

loss���="��        �	���|Yc�A�*

loss_�=�~�       �	<0�|Yc�A�*

loss�=�'��       �	���|Yc�A�*

loss���=H Ѡ       �	h\�|Yc�A�*

lossf��=!o�       �	���|Yc�A�*

loss�->$AT�       �	n��|Yc�A�*

loss�0>qN\`       �	��|Yc�A�*

lossR��=K��b       �	�V�|Yc�A�*

loss̐=7,       �	���|Yc�A�*

loss�vL=D<�g       �	���|Yc�A�*

lossI:Z=F3u       �	�$�|Yc�A�*

loss�	>��(       �	���|Yc�A�*

loss�x>���	       �	LR�|Yc�A�*

loss���=(k�       �	u��|Yc�A�*

loss���=_bP       �	�|�|Yc�A�*

loss;�!>��'       �	;�|Yc�A�*

loss��=��       �	B��|Yc�A�*

lossS�>Ǉ%�       �	�E�|Yc�A�*

loss�k�=���       �	���|Yc�A�*

loss�!_>�33�       �	�o�|Yc�A�*

loss6��=�b��       �	�J�|Yc�A�*

loss��+>��)�       �	h��|Yc�A�*

lossMT�=��'       �	��|Yc�A�*

loss��>B�G       �	g_�|Yc�A�*

loss`��=-J.        �	���|Yc�A�*

loss>Fښ�       �	i��|Yc�A�*

loss�ٹ=�a7�       �	�#�|Yc�A�*

loss�s�=y��8       �	���|Yc�A�*

loss��=ì�       �	�V�|Yc�A�*

loss]b(>�C��       �	j��|Yc�A�*

loss"*>؅ú       �	34�|Yc�A�*

loss{\�<��       �	���|Yc�A�*

loss���=u��l       �	z��|Yc�A�*

loss�X >��^�       �	(D�|Yc�A�*

loss��>�{�       �	��|Yc�A�*

loss��<|lH9       �	���|Yc�A�*

lossLm�=g�e�       �	#,�|Yc�A�*

loss��'>��_       �	���|Yc�A�*

lossx�=����       �	&W�|Yc�A�*

loss �=>�A6D       �	(H�|Yc�A�*

lossOC�=A���       �	��|Yc�A�*

loss��=��)�       �	���|Yc�A�*

loss!N=EB�:       �	�E�|Yc�A�*

loss\��=oCuT       �	L��|Yc�A�*

lossc�=�91�       �	u�|Yc�A�*

lossSS=n @       �	�A�|Yc�A�*

lossߢ�=C�)�       �	���|Yc�A�*

loss��p=��Q       �	�{�|Yc�A�*

lossT�s=&7p�       �	��|Yc�A�*

lossK�=���       �	���|Yc�A�*

loss�!>�b.       �	�r�|Yc�A�*

loss}��=�t8       �	%�|Yc�A�*

loss>
�Lk       �	-��|Yc�A�*

loss��7>n��       �	@h�|Yc�A�*

loss��P>r>%�       �	��|Yc�A�*

loss��=�[U�       �	��|Yc�A�*

loss?��=�8�P       �	.�|Yc�A�*

lossOE�=)J�       �	 ��|Yc�A�*

loss�};>s��	       �	4h�|Yc�A�*

losslh->�݄       �	\9�|Yc�A�*

loss�V>�}L       �	���|Yc�A�*

loss(�W=���       �	Mf�|Yc�A�*

loss�V+>�g�p       �	/��|Yc�A�*

loss�ID=����       �	���|Yc�A�*

loss��>��Q       �	�@�|Yc�A�*

loss��=f��       �	��|Yc�A�*

lossZ�?>^��       �	���|Yc�A�*

lossL#�=�4�       �	�M�|Yc�A�*

loss��=:�Up       �	y��|Yc�A�*

loss��6=T^�6       �	���|Yc�A�*

lossof�=Y2Aa       �	0�|Yc�A�*

loss���=�*�       �	���|Yc�A�*

loss)F�<�ͽ�       �	<��|Yc�A�*

loss��>���       �	�5�|Yc�A�*

loss�
>��"�       �	���|Yc�A�*

loss(�=�qN�       �	��|Yc�A�*

loss��?>N�YI       �	cC�|Yc�A�*

loss��=jV`Z       �	d��|Yc�A�*

loss���=y�       �	��|Yc�A�*

losstp*>�F�       �	D2�|Yc�A�*

loss�;>�>�       �	���|Yc�A�*

loss��>^�W�       �	, }Yc�A�*

loss�C�=�       �	�}Yc�A�*

lossS�>�pU�       �	��}Yc�A�*

loss4�>G�=        �	*R}Yc�A�*

loss�>B��R       �	��}Yc�A�*

loss.�=���#       �	�}Yc�A�*

loss_k�=S�N�       �	�6}Yc�A�*

loss6�P=���       �	��}Yc�A�*

lossʰ>�g{q       �	sh}Yc�A�*

loss���=�D{        �	}Yc�A�*

lossQ�>E��?       �	0�}Yc�A�*

loss���=u�/       �	<}Yc�A�*

losss�>/�H       �	��}Yc�A�*

lossH��<�&��       �	�q}Yc�A�*

loss��=�<ϸ       �	[	}Yc�A�*

lossk�=z�f�       �	�	}Yc�A�*

loss.��=⾡>       �	#L
}Yc�A�*

loss�5R=41�       �	r�
}Yc�A�*

lossS�l>�3Mt       �	0�}Yc�A�*

loss�]�>-M��       �	gE}Yc�A�*

loss�S�=��gM       �	��}Yc�A�*

loss� f>t߹       �	��}Yc�A�*

loss��O=�#��       �	v6}Yc�A�*

lossP>����       �	��}Yc�A�*

loss(�}=Y�       �	{�}Yc�A�*

loss�M=?�       �	�}Yc�A�*

loss�Υ=L��       �	��}Yc�A�*

lossQu�=i�fl       �	8j}Yc�A�*

loss�A">^E.�       �	}Yc�A�*

loss�C�=����       �	}Yc�A�*

loss�R=r�F�       �	��}Yc�A�*

loss2��=� E�       �	�S}Yc�A�*

loss��>�g�       �	��}Yc�A�*

lossw#>TU��       �	��}Yc�A�*

loss2�=x
>^       �	�A}Yc�A�*

loss1��=�L��       �	��}Yc�A�*

loss���=�1w       �	/�}Yc�A�*

loss�� >:���       �	�}Yc�A�*

loss��N>�7mj       �	�}Yc�A�*

loss��=Mr��       �	�f}Yc�A�*

lossj�>y�+}       �	O}Yc�A�*

loss��=%"5       �	o�}Yc�A�*

lossݼ�=^�n       �	�F}Yc�A�*

lossHZ�=y7-       �	��}Yc�A�*

loss���=�@�;       �	�}Yc�A�*

loss�H	>��(       �	 }Yc�A�*

loss�2�=���K       �	��}Yc�A�*

loss�=.���       �	B}Yc�A�*

loss!��=��t!       �	]o }Yc�A�*

loss ?5>.��`       �	!}Yc�A�*

loss���=�G�\       �	կ!}Yc�A�*

loss���=tr�       �	'f"}Yc�A�*

loss�=�߷       �	%t#}Yc�A�*

loss���=�g��       �	�$}Yc�A�*

loss\�=�E��       �	�$}Yc�A�*

loss�%>k	K       �	JE%}Yc�A�*

loss��>l��       �	]�%}Yc�A�*

lossA��=�~'�       �	�z&}Yc�A�*

lossl�=hڣ       �	w'}Yc�A�*

loss���=�ʽ       �	d�'}Yc�A�*

loss{��=B�       �	�b(}Yc�A�*

loss��
>���       �	��(}Yc�A�*

lossm3�=i!��       �	��)}Yc�A�*

loss��=���       �	r2*}Yc�A�*

loss_��=ل��       �	��*}Yc�A�*

lossL��=��A�       �	��+}Yc�A�*

loss��=g5�u       �	�U,}Yc�A�*

lossXN�=��       �	��,}Yc�A�*

loss̳=�t�       �	�-}Yc�A�*

loss�`>[8��       �	�X.}Yc�A�*

lossOϹ=��Ԩ       �	/}Yc�A�*

loss[��=��\\       �	��/}Yc�A�*

loss&qC=��~�       �	L0}Yc�A�*

loss��>%��       �	��0}Yc�A�*

lossJ!U=���!       �	��1}Yc�A�*

lossn��=��I�       �	*r2}Yc�A�*

loss�WW>��M�       �	�3}Yc�A�*

lossa�V=nv�       �	]�3}Yc�A�*

lossX�K=*�r       �	VJ4}Yc�A�*

lossga�=���2       �	��4}Yc�A�*

loss��?>���       �	i�5}Yc�A�*

loss�0�=ՉM/       �	()6}Yc�A�*

loss��#>�"!�       �	��6}Yc�A�*

lossű=>�p�A       �	�c7}Yc�A�*

loss�V�=S��r       �	m�7}Yc�A�*

loss��%>ڙ�R       �	��8}Yc�A�*

lossA4=�e�2       �	49}Yc�A�*

loss��H>�4��       �	��9}Yc�A�*

loss�e>SU֡       �	zr:}Yc�A�*

loss�]>[Һ�       �	�;}Yc�A�*

loss�3S<��δ       �	�<}Yc�A�*

loss��c=o��L       �	�d=}Yc�A�*

loss&�1>,
�z       �	|�>}Yc�A�*

loss3�T>��XJ       �	�B?}Yc�A�*

lossw�`=����       �	��?}Yc�A�*

lossa�=�5��       �	,�@}Yc�A�*

lossC!=�B�       �	?:A}Yc�A�*

loss�ku>�       �	q�A}Yc�A�*

lossQ�)>@h       �	E�B}Yc�A�*

loss��i>�b�       �	�,C}Yc�A�*

loss�M�=�-5       �	��C}Yc�A�*

loss�>�Mp       �	��D}Yc�A�*

loss
i�=��       �	RE}Yc�A�*

loss�8�=Z�       �	W�E}Yc�A�*

lossH�)>E�.:       �	+�F}Yc�A�*

loss�~�=�B�%       �	�;G}Yc�A�*

loss���<�Bؽ       �	��G}Yc�A�*

loss:��=;� 	       �	��H}Yc�A�*

loss`��=�h��       �	z�I}Yc�A�*

loss�O
=�Y=`       �	4J}Yc�A�*

loss\�K=x��{       �	K}Yc�A�*

lossS�=ܚ��       �	��K}Yc�A�*

loss��+=��-.       �	9cL}Yc�A�*

loss��="�       �	�M}Yc�A�*

loss��->�.��       �	��M}Yc�A�*

loss��q>�t�       �	�7N}Yc�A�*

lossF�=����       �	��N}Yc�A�*

lossaF�=���       �	�bO}Yc�A�*

losseI�<��N       �	�3P}Yc�A�*

loss�Ϋ=��h}       �	��P}Yc�A�*

loss���>euI�       �	|Q}Yc�A�*

lossoT�=��>       �	%$R}Yc�A�*

loss̗=��݅       �	��R}Yc�A�*

loss8:�=P߽        �	S}Yc�A�*

loss�~�=�-?       �	�)T}Yc�A�*

loss`��<�[��       �	��T}Yc�A�*

loss3R(=��       �	�zU}Yc�A�*

lossv�>�@�y       �	�*V}Yc�A�*

loss/�
>���       �	t�V}Yc�A�*

loss�9�=���X       �	�rW}Yc�A�*

lossW��=A0=       �	�X}Yc�A�*

loss1m�=�E��       �	��X}Yc�A�*

loss�"�=��Xz       �	TY}Yc�A�*

loss��>�ᐞ       �	��Y}Yc�A�*

loss̪�=��\]       �	��Z}Yc�A�*

lossZ]*=d�Si       �	([}Yc�A�*

loss��<�       �	��[}Yc�A�*

lossd5C>c���       �	�e\}Yc�A�*

loss�/�= @�       �	�,]}Yc�A�*

loss�d=S��       �	3^}Yc�A�*

loss��f=(��       �	8e_}Yc�A�*

loss͸]=-R}�       �	�``}Yc�A�*

loss�T�=��#�       �	�Ca}Yc�A�*

loss��<��L[       �	�6b}Yc�A�*

loss�O>y�J[       �	��b}Yc�A�*

loss��R=O���       �	=�c}Yc�A�*

loss�O�=u_X�       �	d}Yc�A�*

loss��=}��       �	H7e}Yc�A�*

loss� =i��       �	��e}Yc�A�*

loss�3�<�b�       �	�hf}Yc�A�*

loss�=�o�       �	�g}Yc�A�*

loss���=��:&       �	�g}Yc�A�*

lossF�=�U��       �	�dh}Yc�A�*

lossë>�L��       �	�i}Yc�A�*

loss���>�M�       �	t�i}Yc�A�*

loss�l==\��       �	�Vj}Yc�A�*

loss8�= �       �	D�j}Yc�A�*

loss��>h�,       �	Ƥk}Yc�A�*

loss�8}=�p=       �	GWl}Yc�A�*

loss��=(�%�       �	��l}Yc�A�*

loss��=���       �	��m}Yc�A�*

lossD�1>���       �	�Dn}Yc�A�*

lossh�C=bޑh       �	��n}Yc�A�*

lossm>*���       �	��o}Yc�A�*

loss�s�=��       �	�Ap}Yc�A�*

loss)-�=�(k�       �	��p}Yc�A�*

lossR��<��e       �	��q}Yc�A�*

loss� �<�5�e       �	k+r}Yc�A�*

loss�m�<>'�       �	��r}Yc�A�*

loss��=z��	       �	�s}Yc�A�*

loss���=f1       �	�t}Yc�A�*

loss`��<L|��       �	8�t}Yc�A�*

loss���=HXA       �	��u}Yc�A�*

lossQ�#>th�       �	('v}Yc�A�*

loss�.�=<�h       �	��v}Yc�A�*

loss�$ =�+:       �	�w}Yc�A�*

lossl=5�F^       �	�Nx}Yc�A�*

loss�{=e.S�       �	��x}Yc�A�*

lossE��=����       �	�z}Yc�A�*

loss��j=GL�x       �	@�z}Yc�A�*

loss8�8>s�W&       �	�[{}Yc�A�*

loss=6�O�       �	�=|}Yc�A�*

loss��">y���       �	��|}Yc�A�*

lossZ�=�s�       �	�r}}Yc�A�*

loss�/`=����       �	�"~}Yc�A�*

loss���=�/�       �	��}Yc�A�*

loss =�/-�       �	�V�}Yc�A�*

loss�]�=�[�       �	�}Yc�A�*

loss�8=7��D       �	1��}Yc�A�*

loss�(>��       �	.9�}Yc�A�*

loss�D=E��       �	��}Yc�A�*

loss!ɘ=�+~|       �	Y��}Yc�A�*

loss�!=)~�       �	�>�}Yc�A�*

lossDф=F�[       �	�݄}Yc�A�*

loss�K�=�=�       �	�z�}Yc�A�*

loss'�>3���       �	w�}Yc�A�*

lossZl�<�       �	ٳ�}Yc�A�*

loss��=�v�       �	DL�}Yc�A�*

lossfWh=�t�       �	�}Yc�A�*

loss:H�=)��       �	�~�}Yc�A�*

losse�<���6       �	��}Yc�A�*

loss�w�=�ԧb       �	J��}Yc�A�*

loss��>�;D�       �	/P�}Yc�A�*

loss�4_=0��       �	F"�}Yc�A�*

loss�T+=//6�       �	���}Yc�A�*

loss�=�r`       �	�P�}Yc�A�*

loss��"=gQ؆       �	��}Yc�A�*

loss��*=���U       �	���}Yc�A�*

lossOH<@�       �	=H�}Yc�A�*

loss Я<�Q�$       �	z�}Yc�A�*

loss�<R?       �	_~�}Yc�A�*

loss-�b<���       �	 $�}Yc�A�*

loss�R<:���       �	�ʐ}Yc�A�*

lossl�=���       �	�h�}Yc�A�*

lossƛ�;l���       �	0�}Yc�A�*

loss=�;^Љ        �	���}Yc�A�*

loss�m";v)�       �	[�}Yc�A�*

loss�T=,��       �	W�}Yc�A�*

loss?W8>�!ĥ       �	���}Yc�A�*

loss1^�< �1       �	�K�}Yc�A�*

lossX�;<7���       �	1�}Yc�A�*

loss�/=4���       �	�}Yc�A�*

lossԙ?a�C�       �	�B�}Yc�A�*

lossc�(<] �       �	4��}Yc�A�*

loss�g'>���f       �	���}Yc�A�*

lossSxV=K��)       �	MH�}Yc�A�	*

loss�s>,�H       �	%�}Yc�A�	*

loss.��=V�       �	"��}Yc�A�	*

loss��=���s       �	�'�}Yc�A�	*

loss�i>�A�       �	���}Yc�A�	*

loss�>�/{       �	tb�}Yc�A�	*

losse�=�8�N       �	+�}Yc�A�	*

loss���=�Y�       �	���}Yc�A�	*

lossUD�=Z�Ϩ       �	�J�}Yc�A�	*

loss*�>n�)       �	�ߞ}Yc�A�	*

loss�5�= ��5       �	�x�}Yc�A�	*

loss�>z��-       �	��}Yc�A�	*

loss�B9>s���       �	#��}Yc�A�	*

loss���>p��j       �	�l�}Yc�A�	*

lossc�=|��       �	�Ţ}Yc�A�	*

lossz`�=�N�       �	�k�}Yc�A�	*

loss���=s���       �	M.�}Yc�A�	*

loss�¤=����       �	�Ĥ}Yc�A�	*

loss�=jp�       �	�c�}Yc�A�	*

loss�%�=8[
�       �	� �}Yc�A�	*

lossz��=r8�       �	���}Yc�A�	*

lossK@=�k�       �	*;�}Yc�A�	*

loss!��<t�G�       �	p(�}Yc�A�	*

loss�H:=V7�       �	���}Yc�A�	*

loss?�=����       �	X�}Yc�A�	*

lossEx=�|`       �	J�}Yc�A�	*

loss$�)>tA�       �	珪}Yc�A�	*

lossB� >��a       �	0-�}Yc�A�	*

lossɥ�<:�r       �	ͫ}Yc�A�	*

loss��=�÷�       �	�c�}Yc�A�	*

loss�}=0�\       �	n��}Yc�A�	*

loss;�<TSK       �	 ��}Yc�A�	*

loss#��=��+       �	jN�}Yc�A�	*

lossh�=�V�6       �	��}Yc�A�	*

loss�I�<��{d       �	)y�}Yc�A�	*

loss;��=6�P�       �	�E�}Yc�A�	*

loss���=�T�9       �	��}Yc�A�	*

loss�[�=�	�       �	=�}Yc�A�	*

loss��d<���       �	��}Yc�A�	*

lossڛM=V��6       �	K��}Yc�A�	*

loss��=ݕ4�       �	��}Yc�A�	*

loss�=g���       �	�~�}Yc�A�	*

loss	�=]��'       �	D0�}Yc�A�	*

lossL�>e�       �	�е}Yc�A�	*

loss��0>�R<Q       �	Ox�}Yc�A�	*

loss�_�<�       �	��}Yc�A�	*

lossX��=����       �	n��}Yc�A�	*

losst�={V{       �	2=�}Yc�A�	*

loss��8=�t��       �	�Ը}Yc�A�	*

loss��>wNRa       �	g�}Yc�A�	*

loss&
�=��|       �	A�}Yc�A�	*

loss�>/NL       �	נ�}Yc�A�	*

loss@d>��8       �	�8�}Yc�A�	*

lossJK�=h��       �	���}Yc�A�	*

loss�^�<��       �	�g�}Yc�A�	*

loss�{�=L��       �	 ��}Yc�A�	*

loss��=Y�&�       �	q��}Yc�A�	*

loss��->����       �	(�}Yc�A�	*

loss �L>Z%�       �	w��}Yc�A�	*

loss���<c�K.       �	M�}Yc�A�	*

loss��=!��       �	���}Yc�A�	*

loss���=pt�r       �	��}Yc�A�	*

loss�{ >��&�       �	�&�}Yc�A�	*

loss�7�=6Āu       �	j��}Yc�A�	*

loss�T�=�:�       �	�p�}Yc�A�	*

loss@�=�%��       �	Tn�}Yc�A�	*

loss3[�=r�H       �	.�}Yc�A�	*

loss�}�=u��       �	P��}Yc�A�	*

loss��>}s�D       �	IL�}Yc�A�	*

loss���=�o-       �	W>�}Yc�A�	*

loss�1#>o-O�       �	���}Yc�A�	*

loss�O�={���       �	tz�}Yc�A�	*

loss���=���       �	D�}Yc�A�	*

lossf�a=8A7�       �	Ů�}Yc�A�	*

lossIC1=�u}�       �	�D�}Yc�A�	*

losso�>X�ס       �	���}Yc�A�	*

loss��=s���       �	v�}Yc�A�	*

loss}��=�N��       �	m�}Yc�A�	*

loss�>Ҿ��       �	��}Yc�A�	*

lossh:u=?%       �	{K�}Yc�A�	*

loss�,�=9ʧO       �	W��}Yc�A�	*

lossd��=z4*       �	ӆ�}Yc�A�	*

lossȌ�=s���       �	�'�}Yc�A�	*

loss�]>=�;N�       �	���}Yc�A�	*

lossx��=}��3       �	�`�}Yc�A�	*

loss��=ďa�       �	L�}Yc�A�	*

loss ��=X��       �	ӽ�}Yc�A�	*

lossyU�>m���       �	BZ�}Yc�A�	*

loss
o�=�Y��       �	���}Yc�A�	*

lossȖ�=��0�       �	!��}Yc�A�	*

loss���=Z       �	N&�}Yc�A�	*

loss�l�<7�$i       �	���}Yc�A�	*

loss���=}L�-       �	ٙ�}Yc�A�	*

loss�&�=�S�       �	Z��}Yc�A�	*

loss�=&��       �	�L�}Yc�A�	*

loss�=�D�T       �	h��}Yc�A�	*

loss-5�=j       �	D��}Yc�A�	*

lossr�=LӰ�       �	� �}Yc�A�	*

lossb�	<��1       �	���}Yc�A�	*

loss��<)Ƀ�       �	�X�}Yc�A�	*

loss�R�=���       �	
��}Yc�A�	*

loss7��<a��       �	l��}Yc�A�	*

loss`«>\��       �	�n�}Yc�A�	*

loss��R=�}>�       �	��}Yc�A�	*

loss���;��       �	��}Yc�A�	*

lossX<R �{       �	"U�}Yc�A�	*

loss�S�<�O�       �	Z.�}Yc�A�	*

lossF*�=�<E�       �	*��}Yc�A�	*

loss��=[�;�       �	�e�}Yc�A�	*

lossV>�d�       �	���}Yc�A�	*

loss�P:=�1d�       �	���}Yc�A�	*

lossav�<�T��       �	L7�}Yc�A�	*

loss���==��       �	��}Yc�A�	*

loss,,�=C��       �	�q�}Yc�A�	*

lossʷ�=6c��       �	�T�}Yc�A�	*

loss��m=+Ҍ       �	5��}Yc�A�	*

lossM��=��>       �	���}Yc�A�	*

loss�d>����       �	���}Yc�A�	*

loss��=���       �	�=�}Yc�A�	*

loss��=����       �	���}Yc�A�	*

loss�=}�       �	�i�}Yc�A�	*

loss��=����       �	��}Yc�A�	*

lossWq�=�W`�       �	���}Yc�A�	*

lossez=@/��       �	�R�}Yc�A�	*

loss�#i=�9�:       �	, ~Yc�A�	*

lossQ�>�3h7       �	*� ~Yc�A�	*

loss;\>��        �	�B~Yc�A�	*

loss�F�<8u�       �	G�~Yc�A�	*

loss;��=h�4�       �	S{~Yc�A�
*

loss�"�=�[       �	�4~Yc�A�
*

loss��=�-�p       �	E�~Yc�A�
*

loss!D�=�c�^       �	�~Yc�A�
*

loss}/�=���D       �	QN~Yc�A�
*

loss�̃=����       �	+�~Yc�A�
*

loss���<��
�       �	Ú~Yc�A�
*

loss	�>�>8       �	e4~Yc�A�
*

loss�K=��[       �	��~Yc�A�
*

lossA$w=̒�4       �	x~Yc�A�
*

loss��i=L�       �	P	~Yc�A�
*

lossT5U>�3�J       �	h�	~Yc�A�
*

loss>��D       �	zn
~Yc�A�
*

lossn�=���       �	�~Yc�A�
*

loss��=~���       �	��~Yc�A�
*

loss�� >��       �	�6~Yc�A�
*

loss�t�=]�       �	��~Yc�A�
*

lossI��=:.��       �	�e~Yc�A�
*

loss�X�="��&       �	 �~Yc�A�
*

loss��[>Wi�-       �	��~Yc�A�
*

loss�s�=����       �	`V~Yc�A�
*

loss�%�<�R�       �	�~Yc�A�
*

lossM�<ʳE�       �	/�~Yc�A�
*

loss�6)>����       �	�"~Yc�A�
*

lossѐ�=.�d�       �	��~Yc�A�
*

loss#��=!��q       �	�j~Yc�A�
*

loss��>~��o       �	S~Yc�A�
*

loss�:=�A��       �	3�~Yc�A�
*

loss�#8=�mm�       �	�B~Yc�A�
*

lossШ>� �       �	'�~Yc�A�
*

loss���=��bD       �	�t~Yc�A�
*

loss
b�=�S�       �	~Yc�A�
*

losse�=�q�#       �	z�~Yc�A�
*

loss�=��)�       �	�C~Yc�A�
*

loss���<�	��       �	C�~Yc�A�
*

loss��=D�3s       �	�w~Yc�A�
*

lossh��=5*�       �	=~Yc�A�
*

loss|ؘ=��z�       �	��~Yc�A�
*

loss�7:>���       �	�J~Yc�A�
*

loss���=�C       �	z�~Yc�A�
*

loss�8�=�Uv       �	�w~Yc�A�
*

loss �=�b°       �	|~Yc�A�
*

loss�R=�2��       �	D�~Yc�A�
*

lossZ�=>&�       �	�E~Yc�A�
*

loss���=;(��       �	C�~Yc�A�
*

loss�l�<S�.�       �	z�~Yc�A�
*

loss�ȅ=jܡ�       �	��~Yc�A�
*

loss�N>�v       �	BC ~Yc�A�
*

loss�ec=��       �	�� ~Yc�A�
*

lossn�&>��ƛ       �	�{!~Yc�A�
*

lossm6�=��       �	B�"~Yc�A�
*

loss/�=���       �	76#~Yc�A�
*

loss�s=�W �       �	2$~Yc�A�
*

loss�<�=� ��       �	l�$~Yc�A�
*

loss�'�<��ި       �	��%~Yc�A�
*

loss#?>��o'       �	�D&~Yc�A�
*

loss$�@=�W<'       �	�#'~Yc�A�
*

loss�C�=�k|�       �	��'~Yc�A�
*

loss��6=�۰v       �	(~Yc�A�
*

loss��b=�p�c       �	�)~Yc�A�
*

lossEq�<����       �	k�)~Yc�A�
*

loss*��<F�l)       �	YO*~Yc�A�
*

loss6"~=��j�       �	��*~Yc�A�
*

loss��=$�>�       �	=�+~Yc�A�
*

loss��=�!��       �	�,~Yc�A�
*

loss5>��4�       �	k�,~Yc�A�
*

loss:�:=Rg`�       �	2X-~Yc�A�
*

loss)�=f̍n       �	]�-~Yc�A�
*

loss�B=�|��       �	��.~Yc�A�
*

lossS��=*��z       �	�5/~Yc�A�
*

loss��=<��       �	b�0~Yc�A�
*

loss��d=Ĳ�       �	�1~Yc�A�
*

loss�>-�w�       �	)&2~Yc�A�
*

loss]}
>.�       �	%�2~Yc�A�
*

loss��=�ނA       �	#g3~Yc�A�
*

lossȊ�=��0       �	�4~Yc�A�
*

loss}5�<���N       �	1�4~Yc�A�
*

loss�~=�l�D       �	�G5~Yc�A�
*

loss���=vSg�       �	��5~Yc�A�
*

loss�s>FH�       �	�r6~Yc�A�
*

lossā2=+���       �	�7~Yc�A�
*

lossԛ�=�x       �	"�7~Yc�A�
*

loss��=B��       �	�I8~Yc�A�
*

lossE�7=51��       �	��8~Yc�A�
*

loss��<��|       �	it9~Yc�A�
*

loss4��=k(��       �	�:~Yc�A�
*

loss)d6=}ַ�       �	;�:~Yc�A�
*

loss�=���*       �	u:;~Yc�A�
*

loss�=���       �	c�;~Yc�A�
*

lossN��=�Gd�       �	
j<~Yc�A�
*

loss�ٻ=�       �	:=~Yc�A�
*

loss�B�=��9�       �	��=~Yc�A�
*

loss�{<=�i]       �	?>~Yc�A�
*

loss���=ϩF^       �	��>~Yc�A�
*

loss��>4��       �	�n?~Yc�A�
*

loss�P�<���I       �	�@~Yc�A�
*

loss�X=r�;h       �	�@~Yc�A�
*

loss{�=3N-d       �	�uA~Yc�A�
*

loss��`=�.       �	�@B~Yc�A�
*

loss�Z=���       �	:�B~Yc�A�
*

loss(=�[Ġ       �	�C~Yc�A�
*

loss��<$C�       �	6D~Yc�A�
*

loss��=��;       �	��D~Yc�A�
*

loss��Y=5�I�       �	��E~Yc�A�
*

loss7U�<�m�Z       �	�"F~Yc�A�
*

loss;�)=YZV        �	=*G~Yc�A�
*

lossv9=�sy       �	��G~Yc�A�
*

loss�5�=`h�       �	:�H~Yc�A�
*

loss
/�<�e�a       �	�pI~Yc�A�
*

loss6�=�4       �	NJ~Yc�A�
*

lossꦞ=_Sv�       �	2�J~Yc�A�
*

loss�q�=�T��       �	�PK~Yc�A�
*

loss��=�3�.       �	�K~Yc�A�
*

loss�"�<gG�       �	�L~Yc�A�
*

loss/�=�)-       �	�1M~Yc�A�
*

loss���;�~Q�       �	r�M~Yc�A�
*

loss�'F=O�       �	�~N~Yc�A�
*

lossm̵=�l�H       �	�,O~Yc�A�
*

loss�j�=�uP�       �	>�O~Yc�A�
*

lossO_�=*1�       �	�{P~Yc�A�
*

loss�Z=Q�^%       �	�6Q~Yc�A�
*

lossdz�=�F_l       �	��Q~Yc�A�
*

loss�q�<'�n       �	F|R~Yc�A�
*

loss���=�bR�       �		S~Yc�A�
*

lossт=�b��       �	ٵS~Yc�A�
*

loss=B<�%;�       �	�iT~Yc�A�
*

loss_��=i���       �	�U~Yc�A�
*

loss7	>�       �	ѱU~Yc�A�
*

lossl�G=�1��       �	*RV~Yc�A�*

loss%z�=����       �	��V~Yc�A�*

lossk�=��1       �	��W~Yc�A�*

loss��w=�D.�       �	D6X~Yc�A�*

loss؇=p��       �	9�X~Yc�A�*

loss���=�ϱ       �	�oY~Yc�A�*

loss���<���       �	<Z~Yc�A�*

loss���=X��       �	�Z~Yc�A�*

loss�}�=O>_       �	�V[~Yc�A�*

loss5^>����       �	��[~Yc�A�*

loss�a�>����       �	W�\~Yc�A�*

loss�{>���,       �	�<]~Yc�A�*

loss���=��J�       �	x�]~Yc�A�*

loss�H"=��Ƞ       �	u�^~Yc�A�*

lossa\�=qL�       �	�._~Yc�A�*

loss��=��H       �	��_~Yc�A�*

lossM�3>|���       �	�u`~Yc�A�*

lossJ�~<4�n       �	�a~Yc�A�*

loss؅r=|"�       �	��a~Yc�A�*

loss�1�=I��t       �	�b~Yc�A�*

loss�b�=�YMR       �	c'c~Yc�A�*

lossv=�=5�ї       �	S	d~Yc�A�*

lossC�}=}�=�       �	�d~Yc�A�*

loss�z�<�|�H       �	J^e~Yc�A�*

loss���<�F��       �	��e~Yc�A�*

loss���=E��       �	�f~Yc�A�*

loss�ܟ= �i�       �	<g~Yc�A�*

lossd`�=X@k�       �	xh~Yc�A�*

loss��
>+��       �	}%i~Yc�A�*

loss��9=߉�       �	��i~Yc�A�*

loss�5�=(�Y,       �	�_j~Yc�A�*

lossiƌ=��       �	�k~Yc�A�*

loss��g=)Qju       �	�l~Yc�A�*

loss(z�<M�a       �	G�l~Yc�A�*

loss=UA=�-K�       �	0Hm~Yc�A�*

lossc\�=�i`�       �	��m~Yc�A�*

lossQ~�=���       �	�n~Yc�A�*

loss�k=ʧ�       �	�o~Yc�A�*

loss��=���[       �	U�o~Yc�A�*

loss�N�=A���       �	�Vp~Yc�A�*

lossɒ=����       �	��p~Yc�A�*

lossm��=�ЎM       �	�q~Yc�A�*

lossWx�=T�f@       �	�cr~Yc�A�*

lossiӿ=�ZՋ       �	�r~Yc�A�*

loss!��=r�       �	R�s~Yc�A�*

loss��=�!�       �	q9t~Yc�A�*

loss���<c{�4       �	��t~Yc�A�*

loss4�,=^���       �	��u~Yc�A�*

loss_q<�K       �	�Qv~Yc�A�*

loss&y�=���       �	��v~Yc�A�*

losswB\=+��       �	��w~Yc�A�*

loss哧=9�       �	%x~Yc�A�*

loss�<�=XX�       �	�x~Yc�A�*

loss�Xn=V��       �	Yly~Yc�A�*

loss$�`=��q�       �	z~Yc�A�*

lossETL>��v       �	��z~Yc�A�*

lossm(=d9�       �	�}{~Yc�A�*

loss^�=��       �	�Z|~Yc�A�*

loss="�<��1�       �	o�|~Yc�A�*

loss��=mA�:       �	a�}~Yc�A�*

loss�K�=[1�m       �	Sz~~Yc�A�*

lossչ>��       �	~Yc�A�*

loss� �=�v       �	��~Yc�A�*

loss{l[=Xg�       �	�u�~Yc�A�*

loss�W=���o       �	�@�~Yc�A�*

loss���=�Wt       �	d�~Yc�A�*

loss�>>� -       �	V�~Yc�A�*

lossǁ>�/{Q       �	{��~Yc�A�*

lossH�=[�        �	Pp�~Yc�A�*

loss�Q=e�2�       �	��~Yc�A�*

loss_O�=��^       �	���~Yc�A�*

lossT��=tp&z       �	~r�~Yc�A�*

lossa��=��j�       �	}$�~Yc�A�*

loss��#=�X0V       �	�ۇ~Yc�A�*

lossռ=�"iJ       �	���~Yc�A�*

loss ��=]$8       �	y!�~Yc�A�*

loss�0�=6#�       �	ܹ�~Yc�A�*

lossXT>�S       �	XW�~Yc�A�*

loss��=n}4       �	��~Yc�A�*

loss��=OBJ1       �	ё�~Yc�A�*

lossZC=���N       �	<�~Yc�A�*

loss$=���Q       �	��~Yc�A�*

loss4�<��}�       �	=��~Yc�A�*

loss[Ǝ<>�*{       �	�~Yc�A�*

lossf��=�IM�       �	���~Yc�A�*

loss)�<ǜ�W       �	�X�~Yc�A�*

loss��<�n��       �	+��~Yc�A�*

loss��P=���       �	���~Yc�A�*

loss,>�}�       �	�4�~Yc�A�*

loss y=��%_       �	�ԑ~Yc�A�*

loss���=~��       �	�u�~Yc�A�*

loss��>�U��       �	(�~Yc�A�*

loss�<>�AA       �	ҧ�~Yc�A�*

loss-GM={^*       �	�G�~Yc�A�*

loss��=�v]       �	���~Yc�A�*

lossd'�=�ɓ�       �	秕~Yc�A�*

loss#�=O�k�       �	�Y�~Yc�A�*

loss�s�=��D�       �	#��~Yc�A�*

loss\�>>t���       �	���~Yc�A�*

lossC[<�g�       �	SA�~Yc�A�*

lossW�">D�Z�       �	@٘~Yc�A�*

loss�=I�IM       �	_{�~Yc�A�*

loss�Hp=�B]�       �	�~Yc�A�*

lossX�= ��$       �	���~Yc�A�*

loss	}�=)���       �	?T�~Yc�A�*

loss2V�=6,�"       �	�~Yc�A�*

loss�(9=d�e       �	n��~Yc�A�*

loss��#=��9�       �	N�~Yc�A�*

losslm�=�Q̍       �	���~Yc�A�*

loss?��=��       �	禞~Yc�A�*

lossO�#=4g4m       �	�S�~Yc�A�*

loss��>I�?r       �	0��~Yc�A�*

loss@	U=J�u       �	��~Yc�A�*

loss�a=���f       �	�?�~Yc�A�*

losso�0>�u��       �	&8�~Yc�A�*

losss��=.l�       �	zߢ~Yc�A�*

loss�Ў=f*        �	��~Yc�A�*

loss���=�Z��       �	�-�~Yc�A�*

lossv�9>��E�       �	ĥ~Yc�A�*

losse�=�"|       �	�h�~Yc�A�*

loss���<,ל�       �	��~Yc�A�*

loss�@�=X!��       �	��~Yc�A�*

losslk>e��t       �	mU�~Yc�A�*

lossL�%=yu;K       �	5�~Yc�A�*

lossF�)=���       �	�é~Yc�A�*

loss��=	���       �	Ze�~Yc�A�*

loss�w=n9h       �	��~Yc�A�*

loss{b!>�B��       �	f��~Yc�A�*

loss��=�[�9       �	�f�~Yc�A�*

loss��>=�y�       �	{�~Yc�A�*

loss��)=sr!:       �	'ڭ~Yc�A�*

loss���=�ki�       �	䁮~Yc�A�*

loss�A=$��       �	VJ�~Yc�A�*

loss��!<)��D       �	
�~Yc�A�*

lossק="��       �	!��~Yc�A�*

loss��=�z��       �	�N�~Yc�A�*

loss2vp=\3R       �	���~Yc�A�*

loss-�Q>��g       �	p��~Yc�A�*

lossQi�=h$�u       �	m9�~Yc�A�*

loss�<�TU�       �	�ӳ~Yc�A�*

loss� >7��       �	o�~Yc�A�*

loss���=�;gq       �	
�~Yc�A�*

lossv,^=�O5a       �	���~Yc�A�*

loss}�o=�)z�       �	
I�~Yc�A�*

lossS|�<����       �	�~Yc�A�*

loss$��=�-"       �	�з~Yc�A�*

lossMo�=[՗�       �	��~Yc�A�*

loss�.�=��       �	R*�~Yc�A�*

loss1`�<����       �	Թ~Yc�A�*

loss�=9�08       �	0�~Yc�A�*

loss��6=�H��       �	)&�~Yc�A�*

loss]�=���       �	��~Yc�A�*

lossM�Z=��#�       �	��~Yc�A�*

loss3�g=�2ZU       �	�<�~Yc�A�*

loss=ϱ<���       �	Aս~Yc�A�*

loss�z]=n�i       �	�p�~Yc�A�*

loss�>�s��       �	)	�~Yc�A�*

loss�w>���       �	��~Yc�A�*

loss}�=�4       �	K�~Yc�A�*

lossc��=C-�3       �	2"�~Yc�A�*

lossJ�=5��       �	Z��~Yc�A�*

lossh�3<���       �	1^�~Yc�A�*

loss�ռ=eTu�       �	���~Yc�A�*

loss(P�=���       �	t��~Yc�A�*

lossR�[=x#<�       �	�h�~Yc�A�*

losswd >'�       �	,�~Yc�A�*

loss�"�<CG�       �	���~Yc�A�*

loss�?�=�Hr       �	a�~Yc�A�*

loss
�=D�K^       �	���~Yc�A�*

lossL(�=^���       �	~��~Yc�A�*

loss��=l���       �	�)�~Yc�A�*

loss�\`=z��       �	���~Yc�A�*

loss��l=���       �	�[�~Yc�A�*

lossO�=�7�       �	���~Yc�A�*

loss ϥ=򄤥       �	���~Yc�A�*

loss���=���x       �	�5�~Yc�A�*

loss�U=3X�P       �	[��~Yc�A�*

loss��==�Ee       �	���~Yc�A�*

loss��>�8ؐ       �	�;�~Yc�A�*

loss{t�=Ӫ��       �	��~Yc�A�*

loss��=7y%f       �	��~Yc�A�*

loss�'=�v�       �	$D�~Yc�A�*

loss��=�u       �	P��~Yc�A�*

loss�y�=g�W^       �	��~Yc�A�*

lossβ�=��#       �	��~Yc�A�*

loss���=)�4       �	���~Yc�A�*

lossO��=v�?       �		p�~Yc�A�*

loss�7=
��       �	F	�~Yc�A�*

loss7��=H�        �	G��~Yc�A�*

lossa �<���|       �	8M�~Yc�A�*

loss��<tb|�       �	���~Yc�A�*

loss�w�<���       �	���~Yc�A�*

loss,�u=�K:e       �	�6�~Yc�A�*

loss�0=�to       �	M��~Yc�A�*

loss��=����       �	�s�~Yc�A�*

loss��>��I       �	��~Yc�A�*

loss�}$=�;T�       �	���~Yc�A�*

loss�?�=�7�=       �	�S�~Yc�A�*

loss,6�=cy"e       �	���~Yc�A�*

loss}�)>h��C       �	��~Yc�A�*

lossI�=����       �	��~Yc�A�*

lossF)/=[��       �	���~Yc�A�*

loss��=�n'1       �	�I�~Yc�A�*

loss��=��e       �	N��~Yc�A�*

loss��D>�.,�       �	��~Yc�A�*

lossd�C=����       �	t(�~Yc�A�*

loss\4�=��O       �	���~Yc�A�*

lossϔ>�~W       �	�k�~Yc�A�*

loss��>����       �	���~Yc�A�*

lossɑ�<�j��       �	<��~Yc�A�*

lossW��<���[       �	�Y�~Yc�A�*

lossE+�=&���       �	���~Yc�A�*

lossx��=�1D�       �	���~Yc�A�*

loss6"
=�м       �	�;�~Yc�A�*

loss��=�ݠ�       �	���~Yc�A�*

lossD��=�E�r       �	]��~Yc�A�*

losss0>���h       �	�^�~Yc�A�*

loss��=��X       �	X��~Yc�A�*

loss��=v��       �	A��~Yc�A�*

loss�i�=�a�G       �	?T�~Yc�A�*

lossQ��=����       �	���~Yc�A�*

loss��P=%��R       �	���~Yc�A�*

loss�<=Nt�k       �	 �~Yc�A�*

loss�;�=ĖfY       �	h��~Yc�A�*

loss�&t=a�       �	�]�~Yc�A�*

lossC��<z�M�       �	���~Yc�A�*

loss���=1��       �	7��~Yc�A�*

loss�&�=��4�       �	�&�~Yc�A�*

loss��v=I��       �	���~Yc�A�*

loss��<M���       �	�M�~Yc�A�*

lossX�<3[       �	h��~Yc�A�*

lossӫ|<���       �	b��~Yc�A�*

loss]�=ͳ�       �	G�~Yc�A�*

loss1�>�H@�       �	ޮ�~Yc�A�*

loss_�>>�/s       �	[C�~Yc�A�*

loss&��=�q       �	���~Yc�A�*

loss�=}�+       �	S{�~Yc�A�*

loss �'=Z	��       �	��~Yc�A�*

loss�|�=�"	p       �	2��~Yc�A�*

loss�E>�ʅ�       �	tz�~Yc�A�*

loss-��=+u�       �	�#�~Yc�A�*

loss;��=Le(�       �	�;�~Yc�A�*

loss��=3��       �	7��~Yc�A�*

loss��=4�       �	9��~Yc�A�*

loss���=a�z       �	�r�~Yc�A�*

lossd�<:k/       �	�~Yc�A�*

loss/>�c}�       �	� �~Yc�A�*

lossӸB>���       �	��~Yc�A�*

loss%��=���[       �	�.�~Yc�A�*

lossi�>�)5�       �	�9�~Yc�A�*

loss�=[��       �	a��~Yc�A�*

loss�
>=��FL       �	��~Yc�A�*

loss$��=���L       �	�c�~Yc�A�*

loss�B�<	�0�       �	�a�~Yc�A�*

loss!RG<T��6       �	���~Yc�A�*

loss	�<  �       �	�� Yc�A�*

loss쨮=�ѥ�       �	�<Yc�A�*

loss6�G=��s�       �	��Yc�A�*

loss*)=��^�       �	hYc�A�*

loss��5=���       �	pYc�A�*

loss��=�!"�       �	r�Yc�A�*

lossi��=
/��       �	VEYc�A�*

loss��<
Z%Y       �	U�Yc�A�*

loss@,>C�m       �	�pYc�A�*

loss�Q=�(W       �	�Yc�A�*

loss�7�=> �       �	��Yc�A�*

lossC�>�qs'       �	�<Yc�A�*

loss�-�<K�7�       �	��Yc�A�*

loss��=*��       �	niYc�A�*

lossm>=�Y�       �	m�Yc�A�*

loss?��<�<�       �	͔	Yc�A�*

loss�=lK       �	�*
Yc�A�*

loss1�J>g�q�       �	�
Yc�A�*

lossa7P>UG�       �	�aYc�A�*

loss�E�=�N�       �	�Yc�A�*

loss�Oa=A0�6       �	�Yc�A�*

loss��=�u8*       �	�xYc�A�*

loss���= ���       �	�Yc�A�*

loss���<�>�       �	��Yc�A�*

lossBܗ=v�9       �	m�Yc�A�*

lossq��=�.�       �	�FYc�A�*

lossq)=���       �	=�Yc�A�*

loss@jd=i��       �	��Yc�A�*

loss*=#=w��O       �	e9Yc�A�*

loss�=���       �	#�Yc�A�*

loss@�=�"t       �	%tYc�A�*

lossj�6=J�_       �	,Yc�A�*

lossW�<ǐuY       �	"�Yc�A�*

loss1�<�k��       �	�#Yc�A�*

loss��>����       �	�Yc�A�*

loss3#�<z�~�       �	\TYc�A�*

lossM/�<���	       �	E�Yc�A�*

lossfU�=:v�       �	��Yc�A�*

loss�U�=��Ӊ       �	HOYc�A�*

loss��b=t)*j       �	�Yc�A�*

loss2�C=Kw�9       �	��Yc�A�*

loss�v�=�`       �	~WYc�A�*

loss�N=�L�       �	WYc�A�*

lossh�<'��       �	�Yc�A�*

lossM[=P�Y9       �	��Yc�A�*

loss�<=�[N!       �	�6Yc�A�*

loss��=_���       �	?�Yc�A�*

lossΤ�=ABE       �	ŏYc�A�*

loss�`�<�O       �	�. Yc�A�*

lossc�=J6�       �	�� Yc�A�*

loss�Uw<���<       �	io!Yc�A�*

loss4Ū=gf�       �	�1"Yc�A�*

lossX=�B��       �	d�"Yc�A�*

loss��=��V9       �	w#Yc�A�*

loss�m�<���&       �	 $Yc�A�*

loss��a=X��
       �	A�$Yc�A�*

lossHa�<'3(�       �		O%Yc�A�*

lossX�5=Դs       �	R�&Yc�A�*

loss"T�<����       �	4'Yc�A�*

loss�~�=c);        �	��'Yc�A�*

loss(�7<�:�X       �	�a(Yc�A�*

loss�3=�BW�       �	�)Yc�A�*

lossA��=�{�       �	 �)Yc�A�*

loss�G=��       �	qs*Yc�A�*

loss�9"=4rq9       �	�+Yc�A�*

loss�B=w{�       �	b�+Yc�A�*

loss,�\>�\�.       �	�\,Yc�A�*

loss��=F�.�       �	��,Yc�A�*

lossy!=Wv�i       �	M�-Yc�A�*

loss&��=�
�Y       �	�E.Yc�A�*

loss���;�`��       �	O�.Yc�A�*

lossxb<*>�x       �	ƈ/Yc�A�*

loss�i�=�T�\       �	�!0Yc�A�*

loss���<U�ޠ       �	��0Yc�A�*

loss�g;�?�       �	wJ1Yc�A�*

lossv�d<G҈       �	v�1Yc�A�*

loss)�;���       �	��2Yc�A�*

lossS�=Vl�       �	[�3Yc�A�*

lossv�;���:       �	�34Yc�A�*

loss�n<��K�       �	��4Yc�A�*

lossd�};Р�       �	Uk5Yc�A�*

loss�\�<G���       �	�6Yc�A�*

lossRw�=Z��|       �	F�6Yc�A�*

loss���<�M7       �	m:7Yc�A�*

lossd&K;�F9�       �	��7Yc�A�*

loss
�F=���a       �	�p8Yc�A�*

lossX�>9��       �	9&9Yc�A�*

loss�`D<��       �	�9Yc�A�*

lossV�,>�1Ը       �	�V:Yc�A�*

lossO �=��3X       �	��:Yc�A�*

loss�<�=Y�H       �	+�;Yc�A�*

loss���<��58       �	�><Yc�A�*

loss{S�<�r�       �	�I=Yc�A�*

lossL��='�*       �	2�=Yc�A�*

loss�B�=9l��       �	%�>Yc�A�*

loss�Z�=��*i       �	c~?Yc�A�*

loss-�L=
T0       �	�0@Yc�A�*

loss�8�=V<Nw       �	��@Yc�A�*

lossMM�=�T~       �	}�AYc�A�*

lossX��=.?�)       �	'OBYc�A�*

loss6+=��       �	q�CYc�A�*

loss�^�=r�3�       �	pDYc�A�*

loss�!>a��       �	�EYc�A�*

lossJ��<BxQ       �	U�EYc�A�*

loss�ՠ=��Հ       �	�FYc�A�*

loss���=�%�       �	�dGYc�A�*

lossϓO=/��       �	�HYc�A�*

losse�=�Chb       �	_�HYc�A�*

loss~�#=]�k^       �	PTIYc�A�*

loss��|=��b�       �	��IYc�A�*

lossd֓<�:|       �	��JYc�A�*

loss��]<�P"�       �	�KYc�A�*

lossڲ�<6�TZ       �	��KYc�A�*

loss=�B=���q       �	�HLYc�A�*

loss��<y_��       �	S�LYc�A�*

loss�5>]��       �	Y�MYc�A�*

loss�S�=�� �       �	�NYc�A�*

loss�<�<�       �	�NYc�A�*

loss�\=&L��       �	@QOYc�A�*

loss�+�<i��       �	o�OYc�A�*

loss}��<����       �	�PYc�A�*

loss�O=G�Ќ       �	�3QYc�A�*

losse�]=XȔ�       �	��QYc�A�*

loss��<��        �	�rRYc�A�*

lossT=o�tJ       �	7SYc�A�*

loss��=(9>       �	>�SYc�A�*

loss�P�<�6��       �	�iTYc�A�*

loss=�E<)T~s       �	[UYc�A�*

losslX�=D��       �	��UYc�A�*

loss�<#�#�       �	t\VYc�A�*

loss�θ<��XW       �	�WYc�A�*

lossL(�=��2!       �	p�WYc�A�*

loss4ْ=��       �	�NXYc�A�*

loss���=-��        �	��XYc�A�*

loss
8�<_��L       �	�}YYc�A�*

loss��X=���       �	�ZYc�A�*

loss�a�<@{�{       �	�ZYc�A�*

loss��}<艣Z       �	�A[Yc�A�*

loss��=P��       �	�rYc�A�*

loss���=�D��       �	� sYc�A�*

loss� �=ju       �	t�sYc�A�*

lossM��=��+       �	fLtYc�A�*

lossjT=/0��       �	��tYc�A�*

loss�h<��W�       �	�uYc�A�*

loss���=xx��       �	�8vYc�A�*

loss�Y9=^���       �	��vYc�A�*

loss7��=��I       �	8ewYc�A�*

loss��
>De�       �	m�wYc�A�*

loss�KC=|��6       �	e�xYc�A�*

lossG=��+\       �	FyYc�A�*

loss���=��7�       �	��yYc�A�*

loss�>!i�       �	�~zYc�A�*

loss��8=��)       �	e{Yc�A�*

lossE7�=h��       �	!|Yc�A�*

lossm��<�X<�       �	�|Yc�A�*

loss�̱=��1�       �	V�}Yc�A�*

loss�3=��<       �	ۤ~Yc�A�*

loss<�;>��<       �	S�Yc�A�*

loss��<3�       �	!V�Yc�A�*

loss��=Vܯ�       �	D��Yc�A�*

loss��<�a��       �	W��Yc�A�*

lossB�>*���       �	�7�Yc�A�*

loss��!=��@�       �	�؂Yc�A�*

loss��<���       �	By�Yc�A�*

lossEzZ=</�D       �	��Yc�A�*

losstBF=D_�       �	���Yc�A�*

lossnH�=n�w�       �	�V�Yc�A�*

lossrD�=j_D�       �	���Yc�A�*

loss��y=��A�       �	꓆Yc�A�*

loss��3=눬�       �	�6�Yc�A�*

loss?�=PY_       �	6·Yc�A�*

loss:d�='Z�       �	�t�Yc�A�*

loss6.�=�
d�       �	�P�Yc�A�*

loss,�]=)�2       �	���Yc�A�*

loss�<��u       �	㍊Yc�A�*

loss��=ZD�       �	� �Yc�A�*

loss���=�R�:       �	ؼ�Yc�A�*

loss�I�=�y       �		S�Yc�A�*

loss��g=�ӓ�       �	���Yc�A�*

loss�Ļ=<[�K       �	Ȕ�Yc�A�*

lossOi�<2��G       �	�)�Yc�A�*

loss�I�=�?�       �	�͎Yc�A�*

loss�R='�M       �	�o�Yc�A�*

lossX]H>j|�2       �	�Yc�A�*

loss�6=�^�,       �	���Yc�A�*

loss�;C=_��       �	�K�Yc�A�*

loss���=8n��       �	Z/�Yc�A�*

lossb�<p�8       �	�ȒYc�A�*

loss���<�RЁ       �	{h�Yc�A�*

loss�=.��       �	F�Yc�A�*

lossf)�;K0�m       �	 ��Yc�A�*

loss@�_>�?t�       �	�X�Yc�A�*

loss�>/=�wþ       �	���Yc�A�*

loss<*<�g��       �	h��Yc�A�*

loss�<2�}v       �	b0�Yc�A�*

loss&�;��       �	XǗYc�A�*

loss�U=	���       �	�_�Yc�A�*

lossww�=:�U�       �	��Yc�A�*

loss��=�X       �	 ��Yc�A�*

loss��= m?�       �	)@�Yc�A�*

lossc޶<��BK       �	��Yc�A�*

loss�ƀ=%сU       �		��Yc�A�*

loss�'F="^�       �	(+�Yc�A�*

lossWz�<��        �	`ʜYc�A�*

loss ��=C�:       �	l�Yc�A�*

loss��d=���       �	2�Yc�A�*

lossI0>R{��       �	.˞Yc�A�*

lossS�=�cG.       �	�d�Yc�A�*

loss� =β�       �	��Yc�A�*

lossc��=o7G       �	J��Yc�A�*

loss�=2�B�       �	�6�Yc�A�*

loss�U0=؂��       �	�̡Yc�A�*

lossw45=��X       �	
��Yc�A�*

lossd��<5�̨       �	>��Yc�A�*

loss��<��]2       �	 ;�Yc�A�*

lossrg�=0��F       �	Ze�Yc�A�*

loss��=�]�       �	�8�Yc�A�*

loss �=��       �	JӦYc�A�*

loss���=�       �	l�Yc�A�*

lossw�=Db�       �	��Yc�A�*

loss*�m=��E[       �	r��Yc�A�*

loss�'�<$��       �	.�Yc�A�*

losse�=�BC7       �	v��Yc�A�*

lossIUl<�&EX       �	{��Yc�A�*

loss���=���y       �	['�Yc�A�*

lossp)=�D        �	�ͬYc�A�*

loss�=���       �	�a�Yc�A�*

loss�I�<�sX       �	��Yc�A�*

loss�ܑ=DB��       �	:��Yc�A�*

loss@؞=U��       �	�R�Yc�A�*

loss=X=B+�       �	��Yc�A�*

lossM�v=>��       �	���Yc�A�*

loss�9>�,N�       �	�7�Yc�A�*

loss���<����       �	4ױYc�A�*

lossf=V=LS�       �	�w�Yc�A�*

loss�Mu<��K       �	��Yc�A�*

loss�Ʃ=� D       �	�óYc�A�*

loss��a=W�s�       �	�n�Yc�A�*

lossH%I=���0       �	'�Yc�A�*

loss�V�;�^�^       �	I��Yc�A�*

loss��=7�_�       �	�i�Yc�A�*

loss51=��7�       �	N�Yc�A�*

loss$�=&�6�       �	ū�Yc�A�*

loss�=��k�       �	�T�Yc�A�*

loss��<�X&J       �	�Yc�A�*

loss�LA=�� !       �	��Yc�A�*

lossU!>lZ�R       �	:�Yc�A�*

loss=��DJ       �	�ԺYc�A�*

lossrah=cݔ&       �	yy�Yc�A�*

loss�Y=�tFB       �	*�Yc�A�*

lossv�=N=��       �	�|�Yc�A�*

lossq�v<ZyHK       �	�$�Yc�A�*

loss���<�,�(       �	�̾Yc�A�*

loss�N�<J"        �	�ԿYc�A�*

loss�<�=q��a       �	%t�Yc�A�*

loss(�=Tk��       �	�Y�Yc�A�*

loss}��=�1       �	�4�Yc�A�*

lossy�=9�w       �	=
�Yc�A�*

loss��J=0]       �	��Yc�A�*

lossi��=�_       �	���Yc�A�*

loss!�=����       �	m�Yc�A�*

loss�V	=/�p�       �	�.�Yc�A�*

loss��<[�       �	2�Yc�A�*

loss��=0�U�       �	T��Yc�A�*

loss_��=��s       �	-�Yc�A�*

loss�<�<���t       �	$F�Yc�A�*

loss�ŋ="l��       �	���Yc�A�*

loss�r8=��[       �	��Yc�A�*

lossO-�=.W�       �	a��Yc�A�*

lossX3�=�%h       �	F�Yc�A�*

lossn�@=�#�!       �	��Yc�A�*

loss�W<IN��       �	r��Yc�A�*

loss�"�=�Q�       �	���Yc�A�*

lossz&<�a�       �	pz�Yc�A�*

lossd]=��6�       �	,a�Yc�A�*

loss�dD<��9y       �	q��Yc�A�*

loss;�<>.Z�       �	��Yc�A�*

lossą5<��6       �	���Yc�A�*

lossχ�<sz��       �	�3�Yc�A�*

loss֦=�r�$       �	J��Yc�A�*

lossD��=Ŀ5+       �	�l�Yc�A�*

loss��=y�bY       �	��Yc�A�*

loss��=��+}       �	ѱ�Yc�A�*

lossR\0=Sە�       �	O�Yc�A�*

loss[�=        �	��Yc�A�*

loss�p�<��r       �	m<�Yc�A�*

loss�b=J�F       �	���Yc�A�*

lossA��<���       �	Dl�Yc�A�*

loss_��<����       �	�*�Yc�A�*

loss��=i��J       �	%��Yc�A�*

loss ��=���       �	"r�Yc�A�*

loss�l�=���       �	�Yc�A�*

loss���=ｍ       �	 ��Yc�A�*

loss4~�<C��       �	k�Yc�A�*

lossN�X<ũ5/       �	C�Yc�A�*

loss�!>���       �	J��Yc�A�*

lossvc�=lo��       �	1�Yc�A�*

loss�|=��N       �	|��Yc�A�*

loss_Cu=:��       �	��Yc�A�*

lossVe�=��       �	^/�Yc�A�*

loss�F�<�g       �	���Yc�A�*

loss�b�<CDb�       �	�]�Yc�A�*

loss9�=[���       �	�E�Yc�A�*

loss_ �<>A`       �	zr�Yc�A�*

loss��K=��J       �	��Yc�A�*

lossg6�=�uO^       �	Ǟ�Yc�A�*

loss�f<���       �	�6�Yc�A�*

loss�95=_�       �	���Yc�A�*

loss�u=�5a       �	Ad�Yc�A�*

loss4x�<��*       �	d�Yc�A�*

loss<�h<R5��       �	���Yc�A�*

lossqBi=	��       �	>�Yc�A�*

loss "U<�1i       �	��Yc�A�*

loss�_=��;&       �	��Yc�A�*

lossd��=k��       �	+�Yc�A�*

loss*^>�L��       �	n��Yc�A�*

lossO.=��d�       �	2r�Yc�A�*

loss$ߊ<�)9       �	o�Yc�A�*

losssA�;��       �	£�Yc�A�*

loss�r=?	�y       �	JA�Yc�A�*

loss1}6=^��L       �	�
�Yc�A�*

lossC�B<�e�_       �	���Yc�A�*

loss�Y�<�k�       �	�7�Yc�A�*

loss���<E� �       �	���Yc�A�*

loss=A�=�G��       �	���Yc�A�*

loss��I=%�_�       �	9�Yc�A�*

loss��'=h@�q       �	���Yc�A�*

loss��^=S��       �	�v�Yc�A�*

loss+^
>�:K�       �	s�Yc�A�*

loss��l<�G       �	���Yc�A�*

loss�U=с-       �	�<�Yc�A�*

lossT��=�&�K       �	���Yc�A�*

loss̟�;��       �	�f�Yc�A�*

loss�=���!       �	`�Yc�A�*

loss�p=�g��       �	E��Yc�A�*

loss�+o=㽉�       �	�;�Yc�A�*

loss��='�       �	���Yc�A�*

loss�%=��:	       �	�L�Yc�A�*

loss˜�=���8       �	z9�Yc�A�*

loss oW=��M=       �	��Yc�A�*

lossoi:=�+�       �	���Yc�A�*

lossX�T=\�ʈ       �	���Yc�A�*

loss��=�=S       �	z8 �Yc�A�*

loss��f=K��       �	7� �Yc�A�*

loss�Γ=�mD�       �	���Yc�A�*

loss�N>S�vi       �	WB�Yc�A�*

lossװ,=��I       �	���Yc�A�*

lossA=��xq       �	��Yc�A�*

loss�8G=��f1       �	t��Yc�A�*

loss�Z<����       �	Hn�Yc�A�*

lossX%=.��       �	��Yc�A�*

loss�|�<�T$       �	���Yc�A�*

lossi=��eX       �	���Yc�A�*

loss���=0��       �	U/�Yc�A�*

loss݉>��ҽ       �	��Yc�A�*

lossʾ>)gJ       �	=~	�Yc�A�*

loss���=��)�       �	�/
�Yc�A�*

lossZ��=�6�       �	��
�Yc�A�*

lossӈ=m�r       �	�`�Yc�A�*

loss�L�<�R=       �	a��Yc�A�*

loss$�=�G��       �	 ��Yc�A�*

loss&@>R��       �	q=�Yc�A�*

lossx݅<(_w       �	$��Yc�A�*

loss�ˉ=9���       �	�r�Yc�A�*

loss��\=b�@       �	��Yc�A�*

loss���<HH>�       �	��Yc�A�*

loss�2>=a��       �	�V�Yc�A�*

lossv�<XT�       �	���Yc�A�*

loss\ʄ<�$�1       �	���Yc�A�*

lossy<�=�O       �	,+�Yc�A�*

loss5O�=6���       �	��Yc�A�*

loss��=�I�       �	��Yc�A�*

loss�;>{��       �	5�Yc�A�*

loss�=�@�       �	���Yc�A�*

loss��'=�2�6       �	�e�Yc�A�*

lossv��=��`       �	x�Yc�A�*

lossơ=��i�       �	G��Yc�A�*

loss1D;-U�       �	�D�Yc�A�*

lossW\?=�lM       �	e��Yc�A�*

loss�я=���       �	���Yc�A�*

loss��=3
��       �	��Yc�A�*

loss��)=A�eI       �	��Yc�A�*

loss���=r��       �	��Yc�A�*

loss�$=G��       �	��Yc�A�*

loss��<��#�       �	���Yc�A�*

loss]/=}	��       �	�E�Yc�A�*

loss_��=�L�t       �	@��Yc�A�*

loss�AQ=a�|�       �	Kx�Yc�A�*

loss�J�=J,,�       �	��Yc�A�*

lossO@=�ѥ       �	n��Yc�A�*

loss�Ĩ=��X�       �	�6�Yc�A�*

lossn3�;^L�       �	���Yc�A�*

loss���<F�0       �	�e �Yc�A�*

lossr��<Y�?       �	G!�Yc�A�*

lossi�=����       �	��!�Yc�A�*

loss�p|=�b\�       �	 u"�Yc�A�*

loss�<=��5       �	�#�Yc�A�*

loss ��=T��       �	�#�Yc�A�*

lossi��=�:       �	�4$�Yc�A�*

loss;�'=�,4       �	9�$�Yc�A�*

loss�v�='@�       �	˽%�Yc�A�*

loss�_�=�1�       �	8�&�Yc�A�*

lossV��=C\
�       �	�W'�Yc�A�*

lossL��<x�(�       �	�'�Yc�A�*

loss�ԁ=��~       �	m�(�Yc�A�*

loss��<��L�       �	�")�Yc�A�*

loss��=�-��       �	x�)�Yc�A�*

lossa#�<�I��       �	MN*�Yc�A�*

loss��<�%�	       �	�*�Yc�A�*

loss� ==#� �       �	܄+�Yc�A�*

lossw�<U#��       �	�$,�Yc�A�*

lossjJ�=�4�       �	��,�Yc�A�*

loss���=Z�:I       �	�[-�Yc�A�*

loss���=��)5       �	��-�Yc�A�*

loss ��<*<�       �	�.�Yc�A�*

loss�"=�7t       �	�A/�Yc�A�*

loss�=���       �	�/�Yc�A�*

loss��?=I0�       �	ގ0�Yc�A�*

lossR�<&�       �	�01�Yc�A�*

lossZ�=���L       �	��1�Yc�A�*

loss1={R��       �	;�2�Yc�A�*

loss.�)=	./c       �	+3�Yc�A�*

lossی=MI��       �	z�3�Yc�A�*

loss��=Z�\}       �	�f4�Yc�A�*

loss6��=_v�       �	;5�Yc�A�*

loss
��=�rĂ       �	�5�Yc�A�*

lossR~<� �       �	%A6�Yc�A�*

lossc��<!�       �	^�6�Yc�A�*

lossA<�64       �	�t7�Yc�A�*

loss#�=��z       �	�8�Yc�A�*

loss�I�;i��(       �	"�8�Yc�A�*

lossعz<�g�       �	�D9�Yc�A�*

loss��D<�5�'       �	h�9�Yc�A�*

loss�!>�[��       �	��:�Yc�A�*

loss<S=ѧ�       �	�;�Yc�A�*

loss��>}  R       �	��;�Yc�A�*

lossQݼ=ބsk       �	2U<�Yc�A�*

loss|�=)���       �	��<�Yc�A�*

loss��<"}�       �	E�=�Yc�A�*

loss?�s=/���       �	@�>�Yc�A�*

loss^� =���=       �	(d?�Yc�A�*

loss�=p="_:�       �	7@�Yc�A�*

loss��M=ϐ��       �	��@�Yc�A�*

loss���=��v       �	d[A�Yc�A�*

loss8[<p�ҥ       �	�B�Yc�A�*

loss,��=��[)       �	֭B�Yc�A�*

loss�C�<05̫       �	}�C�Yc�A�*

loss���=�	�U       �	k�D�Yc�A�*

loss3)�<V�k       �	�E�Yc�A�*

loss�Sh=����       �	�F�Yc�A�*

loss��=��'       �	/G�Yc�A�*

loss�=Y.�       �	��G�Yc�A�*

loss ͱ<�C
�       �	�I�Yc�A�*

loss熈=P�=       �	d�I�Yc�A�*

loss權<�g�_       �	QJ�Yc�A�*

loss��`=��P       �	��J�Yc�A�*

loss=�=l��<       �	�K�Yc�A�*

loss>ˣ=��T       �	�L�Yc�A�*

loss�0<�&t�       �		OM�Yc�A�*

loss��=|*��       �	��M�Yc�A�*

lossw�m=ԧk�       �	��N�Yc�A�*

loss�6�<���M       �	6vO�Yc�A�*

loss3P�=�Nا       �	P�Yc�A�*

loss �=dT}�       �	\�P�Yc�A�*

loss)��<Z�2#       �	�SQ�Yc�A�*

lossR�+=���       �	��Q�Yc�A�*

loss���=-�m       �	%�R�Yc�A�*

loss0s�=T#ۆ       �	=+S�Yc�A�*

loss��<�G
�       �	G�S�Yc�A�*

loss5�=�)�4       �	�eT�Yc�A�*

lossa��</���       �	!U�Yc�A�*

loss �+=�2 N       �	��U�Yc�A�*

loss��=���j       �	�7V�Yc�A�*

loss�3=x[��       �	��V�Yc�A�*

loss�ڑ=��l�       �	@�W�Yc�A�*

loss�B�<!�\r       �	�\X�Yc�A�*

loss��=f��       �	��X�Yc�A�*

lossƼ�<�ؖ       �	]Z�Yc�A�*

loss��=��sY       �	<�Z�Yc�A�*

lossY=��ED       �	�b[�Yc�A�*

lossE��<yyo       �	�\�Yc�A�*

lossᶎ=���       �	�\�Yc�A�*

loss���='KG       �	FC]�Yc�A�*

loss��,>���7       �	��]�Yc�A�*

lossL)[<)�       �	�z^�Yc�A�*

loss��= `��       �	EH_�Yc�A�*

loss��p=���O       �	��_�Yc�A�*

lossV
=~]��       �	��`�Yc�A�*

lossa�<��!�       �	�Ta�Yc�A�*

losst��<N2(       �	{b�Yc�A�*

loss^F=ƧS/       �	��b�Yc�A�*

loss���=4�Q�       �	7Oc�Yc�A�*

loss� �=8�I�       �	�c�Yc�A�*

lossZV=	��       �	;�d�Yc�A�*

lossM �<��ſ       �	
.e�Yc�A�*

loss]�<e�Y       �	A�e�Yc�A�*

loss@s�=l�)�       �	�rf�Yc�A�*

lossn�?= ]       �	pg�Yc�A�*

loss��=?b�?       �	[�g�Yc�A�*

loss�4�<��       �	�Sh�Yc�A�*

loss�s,=�o��       �	��h�Yc�A�*

lossV�N>W�[       �	o�i�Yc�A�*

loss�=�=@�FK       �	.rj�Yc�A�*

loss=�=�=       �	�k�Yc�A�*

lossDt= 2x�       �	ڬk�Yc�A�*

losshlB=�b-%       �	�Kl�Yc�A�*

loss�<V3�       �	-�l�Yc�A�*

loss��=1��T       �	k�m�Yc�A�*

loss�T�=�z��       �	q;n�Yc�A�*

loss1�<��t�       �	k�n�Yc�A�*

loss%��=��6       �	Ulo�Yc�A�*

loss��<a�S-       �	Gp�Yc�A�*

loss-�=+%��       �	��p�Yc�A�*

lossR�=�'	       �	(Gq�Yc�A�*

loss}��=��v       �	��q�Yc�A�*

loss#S�=c�e0       �	P�r�Yc�A�*

loss�z=�w       �	~5s�Yc�A�*

loss6�>��       �	A�s�Yc�A�*

loss�FU=r���       �	�tt�Yc�A�*

loss|k=����       �	u�Yc�A�*

lossf>��$       �	t�u�Yc�A�*

loss��8=���4       �	Zv�Yc�A�*

lossZ��<IN�5       �	T�v�Yc�A�*

loss�@|=����       �	ޫw�Yc�A�*

loss�ƅ=��X�       �	�Px�Yc�A�*

loss���=~�&�       �	N�x�Yc�A�*

loss��<U:}�       �	t�y�Yc�A�*

loss�`=�>�       �	m;z�Yc�A�*

lossiG=	���       �	��z�Yc�A�*

loss	�=�QT�       �	�{�Yc�A�*

loss�/�=%�p       �	RC|�Yc�A�*

loss��<���^       �	�}�Yc�A�*

loss��-=��"D       �	��}�Yc�A�*

loss$�=cT�       �	�o~�Yc�A�*

loss�>�<��       �	J|�Yc�A�*

loss�@=�� *       �	Y��Yc�A�*

loss���<�$��       �	$���Yc�A�*

loss�s=��f       �	�P��Yc�A�*

lossx>=���       �	����Yc�A�*

loss%:�=�k�=       �	����Yc�A�*

loss��t=ݙ]�       �	�0��Yc�A�*

loss�s�<��`       �	F̓�Yc�A�*

loss��,=��-       �	e��Yc�A�*

loss��	=�_��       �	"���Yc�A�*

lossfJ�=׹?�       �	����Yc�A�*

loss�c�=����       �	"5��Yc�A�*

loss��<���Q       �	<���Yc�A�*

loss�Σ=�OX�       �	#���Yc�A�*

lossM�=�6��       �	�Q��Yc�A�*

loss �=ԽR       �	��Yc�A�*

loss�Ф=&=��       �	=҉�Yc�A�*

loss�a�=��G�       �	�l��Yc�A�*

loss�5�=����       �	�	��Yc�A�*

loss��=D=�x       �	?���Yc�A�*

loss��<�+�H       �	�F��Yc�A�*

loss��>=����       �	�Yc�A�*

loss��=����       �	:���Yc�A�*

loss�=Z�'x       �	:=��Yc�A�*

loss<d*=��$�       �	T掀Yc�A�*

lossܿ=|        �	Ŏ��Yc�A�*

loss}�<}f�6       �	�6��Yc�A�*

loss���=�mļ       �	NՐ�Yc�A�*

loss�(�=�r6%       �	�v��Yc�A�*

loss�/1>/f7�       �	�"��Yc�A�*

loss��=�I�!       �	�Œ�Yc�A�*

loss�$�=Hwl       �	�c��Yc�A�*

loss�F�=��X�       �	���Yc�A�*

loss��=Jܤ�       �	s���Yc�A�*

loss>+=���       �	ta��Yc�A�*

loss�Fp=y��4       �	=��Yc�A�*

loss&�%=���       �	����Yc�A�*

lossJ��=˧       �	y]��Yc�A�*

loss��=m1�       �	. ��Yc�A�*

lossf��<��       �	�"��Yc�A�*

loss��S<��*|       �	����Yc�A�*

loss}eI<�4e�       �	'���Yc�A�*

loss�k�<�ɞ�       �	B`��Yc�A�*

loss!x<Ҳ��       �	/���Yc�A�*

loss���=w�c�       �	���Yc�A�*

loss�e�=��}+       �	&7��Yc�A�*

loss<!�=��"       �	�ѝ�Yc�A�*

loss���=�1|�       �	����Yc�A�*

loss�4q<��q�       �	2��Yc�A�*

loss}�=��l       �	�ݠ�Yc�A�*

lossQ�S>|+j�       �	${��Yc�A�*

lossZ}=[�VW       �	���Yc�A�*

loss?�o<���I       �	`Z��Yc�A�*

lossS|r=$��       �	U���Yc�A�*

loss��=%���       �	
���Yc�A�*

loss��B=m�>e       �	攥�Yc�A�*

loss���<G�        �	7��Yc�A�*

loss;0	>���:       �	�զ�Yc�A�*

loss`�=:_4�       �	���Yc�A�*

loss�n=ȟk       �	˨�Yc�A�*

loss�H)=�o       �	�ᩀYc�A�*

lossD�=&Orz       �	�~��Yc�A�*

loss,�Q=��o       �	���Yc�A�*

lossq�1=��       �	�ѫ�Yc�A�*

lossa��=�D�       �	�l��Yc�A�*

loss)ߚ<8ik�       �	��Yc�A�*

loss��X=`$w�       �	����Yc�A�*

loss��=�       �	����Yc�A�*

loss%�=�O�       �	1@��Yc�A�*

lossjf=b_�?       �	�寀Yc�A�*

lossx�*<i�f       �	9���Yc�A�*

loss�}�<�X��       �	���Yc�A�*

lossj>�=�̃       �	@���Yc�A�*

loss�@=>![       �	ZI��Yc�A�*

loss$� >>|�_       �	d곀Yc�A�*

lossMC�<C(       �	񜴀Yc�A�*

lossƩ>=��A       �	�:��Yc�A�*

lossDl�=-���       �	�ٵ�Yc�A�*

loss��=9L
�       �	�u��Yc�A�*

loss�Bc=o�%       �	,��Yc�A�*

loss�?�<V�c
       �	����Yc�A�*

loss&(H=N�$       �	F��Yc�A�*

lossqpv<v7�       �	�߸�Yc�A�*

loss1p�=b5�@       �	�w��Yc�A�*

lossQ�>�\:�       �	�I��Yc�A�*

loss�	�=/`~       �	�躀Yc�A�*

loss��=8��       �	I���Yc�A�*

loss� �=M���       �	!��Yc�A�*

loss]'=��H       �	���Yc�A�*

lossX^=wM7F       �	�N��Yc�A�*

lossޅ=�t=       �	t콀Yc�A�*

loss��=-yM"       �	�.��Yc�A�*

loss	�<�9�C       �	�Կ�Yc�A�*

lossV��=����       �	�u��Yc�A�*

loss�`,=nɘI       �	�`��Yc�A�*

loss�A=S-�@       �	c|Yc�A�*

losse��<�g׼       �	��ÀYc�A�*

loss�(�=��/G       �	NDĀYc�A�*

lossO6=�m�*       �	��ŀYc�A�*

lossh.=��#�       �	�,ƀYc�A�*

loss_��=,OZ�       �	!ǀYc�A�*

loss��j=I�L&       �	A�ǀYc�A�*

lossα=�Nr       �	xȀYc�A�*

loss$̠=Cp))       �	�ɀYc�A�*

losse��=����       �	��ɀYc�A�*

loss�1
=��^>       �	�_ʀYc�A�*

loss���<�!�h       �		ˀYc�A�*

loss�ޅ=���       �	V�ˀYc�A�*

loss�Z�=��G       �	aR̀Yc�A�*

loss�8�<�	=       �	�̀Yc�A�*

lossL�=�8W�       �	��̀Yc�A�*

loss_i�<i���       �	�@΀Yc�A�*

loss�+�=k7�F       �	��΀Yc�A�*

loss���=b��?       �	SyπYc�A�*

lossc1�<�x�{       �	(ЀYc�A�*

loss�;_=�EI       �	�ЀYc�A�*

lossQ��<���       �	itрYc�A�*

loss�=1M�       �	{ҀYc�A�*

lossM�Q=UK?v       �	��ҀYc�A�*

loss�A�=�Ҋ       �	F]ӀYc�A�*

loss�#=�>       �	`ԀYc�A�*

loss��*=�Ƃq       �	=�ԀYc�A�*

loss�}�<4�%�       �	<ՀYc�A�*

lossF �=��t�       �	��ՀYc�A�*

loss�`�<����       �	�lրYc�A�*

lossH;�<�Hr       �	�
׀Yc�A�*

loss���<�/�       �	ʣ׀Yc�A�*

loss��\=�ZI�       �	y;؀Yc�A�*

lossB^=d5?       �	��؀Yc�A�*

loss�3�<J��}       �	�|ـYc�A�*

loss/��=O���       �	>%ڀYc�A�*

loss�H�<=���       �	�ڀYc�A�*

loss9�=�]�       �	fkۀYc�A�*

lossSh<_�*O       �	�܀Yc�A�*

lossc��<�7|�       �	��܀Yc�A�*

loss�0�<���       �	�ހYc�A�*

lossWn><oB       �	;�ހYc�A�*

loss�CQ;��e       �	�h߀Yc�A�*

loss�ϧ<d�ܻ       �	�:��Yc�A�*

loss�K�;�GZ�       �	0-�Yc�A�*

loss�8%=Z��G       �	^��Yc�A�*

loss*��<jT�D       �	t��Yc�A�*

loss��;4�a�       �	\8�Yc�A�*

loss@�<8�Yh       �	���Yc�A�*

loss1��;a3c{       �	�x�Yc�A�*

loss��:�_�       �	�Yc�A�*

loss�>9:e��       �	���Yc�A�*

loss˶<5m��       �	rS�Yc�A�*

loss��!=�ܯ       �	H4�Yc�A�*

loss<s=��       �	$��Yc�A�*

lossT��:�?��       �	���Yc�A�*

loss��H=�	       �	��Yc�A�*

loss�dp>���       �	=a�Yc�A�*

loss��;ǐ�       �	F#�Yc�A�*

loss��F>6�@       �	��Yc�A�*

loss=Z�<�3�       �	�n�Yc�A�*

lossú�=ηw�       �	(�Yc�A�*

lossU�=3��       �	���Yc�A�*

loss��<�-"�       �	rQ�Yc�A�*

lossi+	>�/G�       �	5��Yc�A�*

loss�h�<�̲�       �	��Yc�A�*

loss4��<��       �	k(��Yc�A�*

lossv�'=f?       �	=�Yc�A�*

loss�p1=���-       �	��Yc�A�*

loss� >�$;       �	Q��Yc�A�*

lossR��=bC�       �	Н�Yc�A�*

loss4t�=�0�       �	�8�Yc�A�*

loss�Ħ=y��D       �	k��Yc�A�*

loss�6	>��?�       �	 s��Yc�A�*

loss̨=#Ҩo       �	j��Yc�A�*

loss��=!��+       �	����Yc�A�*

losst��=@��       �	�\��Yc�A�*

loss�9=S�       �	����Yc�A�*

loss:�<T�U       �	f���Yc�A�*

loss;=W �       �	�A��Yc�A�*

loss�i=�m-�       �	����Yc�A�*

loss��<�52J       �	)���Yc�A�*

loss�F�<���       �	K=��Yc�A�*

loss��><�'�j       �	����Yc�A�*

loss!4V=�QJ       �	�}��Yc�A�*

loss�K<�BRu       �	d!��Yc�A�*

lossO[�=B9�       �	����Yc�A�*

loss1I�=�;FY       �	;���Yc�A�*

loss2��<�d�       �	Aa��Yc�A�*

loss8P=�>�>       �	9 �Yc�A�*

loss���<�B�       �	�� �Yc�A�*

lossO�%;&��       �	�Q�Yc�A�*

loss���<��
       �	}�Yc�A�*

lossM�;Q�m�       �	��Yc�A�*

lossfc7=Sz��       �	�U�Yc�A�*

loss1��=�8Uq       �	���Yc�A�*

lossl��=Yl]�       �	P��Yc�A�*

lossi��=�1|       �	�f�Yc�A�*

loss���;�9'       �	.�Yc�A�*

loss�}�<G��       �	Ǟ�Yc�A�*

lossI�<�B�       �	VD�Yc�A�*

loss׀>@귂       �	���Yc�A�*

lossH�=[R��       �	t��Yc�A�*

loss;ek=ķ       �	�C	�Yc�A�*

loss6 �=���K       �	 �	�Yc�A�*

lossS��<hV�;       �	��
�Yc�A�*

loss�[=��	�       �	�<�Yc�A�*

loss���<����       �	 ��Yc�A�*

loss%4�<�9�]       �	j��Yc�A�*

losst��=���       �	A$�Yc�A�*

loss��=c2�       �	��$�Yc�A�*

lossh�*>]($�       �	;q%�Yc�A�*

lossF��=�n�       �	�&�Yc�A�*

loss2�=0!��       �	�&�Yc�A�*

loss'�=�y�       �	�d'�Yc�A�*

loss�-a=�G�       �	L�'�Yc�A�*

loss��=�x�P       �	%�(�Yc�A�*

lossSy>��
�       �	jP)�Yc�A�*

loss幭=����       �	Q�)�Yc�A�*

loss 7=ԫs       �	��*�Yc�A�*

loss.�W=X:m�       �	�:+�Yc�A�*

loss�>W�g       �	��+�Yc�A�*

loss�Ϥ=?�4       �	��,�Yc�A�*

losseP=3��:       �	� -�Yc�A�*

loss�=�Rg�       �	7�-�Yc�A�*

loss��<p_�7       �	��.�Yc�A�*

loss{�=�.H       �	�)/�Yc�A�*

loss͇=��iq       �	��/�Yc�A�*

loss�{�=rݯ       �	To0�Yc�A�*

loss4��=�v�l       �	M1�Yc�A�*

lossh�+>�0       �	!�1�Yc�A�*

loss�;�<��}.       �	�I2�Yc�A�*

lossOM�=o��!       �	O�2�Yc�A�*

lossF,=3�f�       �	f�3�Yc�A�*

loss�9
=���       �	�"4�Yc�A�*

loss/ci<��       �	*�4�Yc�A�*

loss�N�<g��0       �	"n5�Yc�A�*

lossJ�=x�       �	�6�Yc�A�*

loss��<����       �	&�6�Yc�A�*

lossn�<uj0R       �	�E7�Yc�A�*

loss=	���       �	7�7�Yc�A�*

lossnN@=a���       �	�{8�Yc�A�*

loss�&�=2|a       �	!"9�Yc�A�*

loss㋓<:.
       �	��9�Yc�A�*

loss@w=Z_p�       �	�^:�Yc�A�*

losset=W�       �	��:�Yc�A�*

loss��=Rm�l       �	<�;�Yc�A�*

loss�(>	�O=       �	t@<�Yc�A�*

loss��=Eo       �	��<�Yc�A�*

loss�$(=4�E�       �	H�=�Yc�A�*

lossJA�=8X�       �	�">�Yc�A�*

loss[@2<V       �	t?�Yc�A�*

loss=�'=&8�       �	&@�Yc�A�*

loss��A=���       �	��@�Yc�A�*

lossֹF=�9�       �	@jA�Yc�A�*

loss�=X�M       �	�hB�Yc�A�*

loss��s=�š{       �	zC�Yc�A�*

lossxbA<���w       �	�C�Yc�A�*

loss�2<����       �	9�D�Yc�A�*

loss�<���       �	��E�Yc�A�*

losszP�=fp+�       �	ԀF�Yc�A�*

loss9=	�`       �	='G�Yc�A�*

loss��/>�8�       �	8�G�Yc�A�*

loss���=�א=       �	�H�Yc�A�*

lossSe_;�}       �	�,I�Yc�A�*

loss�ɉ;J��       �	��I�Yc�A�*

loss�0<z�p       �	�tJ�Yc�A�*

loss�)4=�_{c       �	sK�Yc�A�*

lossZ%=28Y       �	��K�Yc�A�*

loss�:>p�
�       �	�TL�Yc�A�*

lossMn;=&�,�       �	��L�Yc�A�*

loss1"=��&�       �	ٔM�Yc�A�*

lossJr�=��v       �	�3N�Yc�A�*

lossߖ�<�`@�       �	��N�Yc�A�*

loss��=W��{       �	oO�Yc�A�*

loss���<���       �	�P�Yc�A�*

loss��<*���       �	�P�Yc�A�*

loss(��=(�+c       �	�RQ�Yc�A�*

loss�6!=%>��       �	��Q�Yc�A�*

loss$��=;#�       �	��R�Yc�A�*

lossw�q=�=c       �	�!S�Yc�A�*

loss���=ǌ�       �	s�S�Yc�A�*

loss��=��       �	�eT�Yc�A�*

loss��<��iu       �	�U�Yc�A�*

loss{�F<KrǓ       �	�U�Yc�A�*

loss\X�<���       �	@V�Yc�A�*

loss7�=k�1�       �	��V�Yc�A�*

loss��<�\
�       �	�{W�Yc�A�*

lossd��<?Sb�       �	�X�Yc�A�*

loss�J3=Ӄ��       �	�X�Yc�A�*

loss��=`��M       �	�OY�Yc�A�*

loss���=q߈�       �	l�Y�Yc�A�*

loss��<lZ��       �	��Z�Yc�A�*

lossR��<8M       �	l#[�Yc�A�*

loss���<��       �	��[�Yc�A�*

lossQhJ=���1       �	/k\�Yc�A�*

loss��|=�:S       �	 
]�Yc�A�*

loss}6=�x˵       �	D�]�Yc�A�*

loss���<�
�       �	�=^�Yc�A�*

loss�4�= aq�       �	��^�Yc�A�*

loss���<Mj�_       �	_y_�Yc�A�*

loss\��= G�       �	U`�Yc�A�*

loss;0�<a�+�       �	��`�Yc�A�*

loss�= H�       �	Jba�Yc�A�*

loss�ο<��8�       �	��a�Yc�A�*

loss�X�=�{�7       �	��b�Yc�A�*

loss낅=h.�       �	+5c�Yc�A�*

loss7b=XF^       �	��c�Yc�A�*

loss2��<�\�m       �	��d�Yc�A�*

loss�B�<�t-�       �	�4e�Yc�A�*

lossN<8<
��F       �	z�e�Yc�A�*

loss�?k=�S�|       �	�xf�Yc�A�*

loss�F=oH��       �	�g�Yc�A�*

loss��3=l7��       �	�g�Yc�A�*

loss	[�=����       �	�Gh�Yc�A�*

loss���<$UW�       �	��h�Yc�A�*

lossV�q<.��       �	w�i�Yc�A�*

loss�N�=��{6       �	�j�Yc�A�*

loss3G= �`.       �	��j�Yc�A�*

loss3�>��D       �	�Gk�Yc�A�*

loss#�X<�`��       �	C�k�Yc�A�*

loss.5=[�Td       �	}l�Yc�A�*

lossH��<���6       �	�m�Yc�A�*

loss!��<D��2       �	��m�Yc�A�*

loss6l=<慧       �	�Nn�Yc�A�*

loss���=��G       �	�*o�Yc�A�*

loss�!�=V"��       �	��o�Yc�A�*

loss<K=��1�       �	�[p�Yc�A�*

loss��E=��       �	EHq�Yc�A�*

loss���=m��t       �	:r�Yc�A�*

loss�g�=����       �	��r�Yc�A�*

loss��t=;.�       �	aqs�Yc�A�*

loss��=�L�$       �	�t�Yc�A�*

loss@W�;�U4�       �	�t�Yc�A�*

loss��<Q޸:       �	|Fu�Yc�A�*

loss��=EIz!       �	p�u�Yc�A�*

loss�;�<Qk�       �	�v�Yc�A�*

loss��=�6�       �	�*w�Yc�A�*

loss�]o=7��       �	��w�Yc�A�*

loss��$< d�       �	zx�Yc�A�*

loss��<O�_�       �	[	z�Yc�A�*

loss]=���       �	��z�Yc�A�*

loss�/G<U��       �	�a{�Yc�A�*

loss�>="p8w       �	� |�Yc�A�*

loss:E9=;83       �	k�|�Yc�A�*

loss�=J��       �	F?}�Yc�A�*

loss�Zt;,�       �	 �}�Yc�A�*

loss��9<�F       �	;�~�Yc�A�*

loss�/�<4J8       �	t(�Yc�A�*

loss<qO<��TF       �	�,��Yc�A�*

loss%�=�(a       �	˼��Yc�A�*

loss(,&=:P�       �	�P��Yc�A�*

loss���=��c�       �	聁Yc�A�*

loss���=��       �	[���Yc�A�*

loss�Y�=e�b�       �	�V��Yc�A�*

lossཀ=���9       �	�
��Yc�A�*

loss��<�Ǻ9       �	�ۄ�Yc�A�*

loss���<)�T;       �	:ʅ�Yc�A�*

loss#g�<�R]       �	-���Yc�A�*

loss�W^=�ޞP       �	5@��Yc�A�*

loss�/�=�:�M       �	�և�Yc�A�*

lossÉ�=��A�       �	�y��Yc�A�*

loss���<��h       �	���Yc�A�*

lossN�=�3&       �	���Yc�A�*

loss?~�<Ul�       �	iQ��Yc�A�*

loss�
�=�1xF       �	�抁Yc�A�*

loss�7�<���C       �	����Yc�A�*

loss��=���Y       �	:��Yc�A�*

loss��<i�ĭ       �	f���Yc�A�*

loss�Y�<���       �	h^��Yc�A�*

loss�e�=�vh       �	����Yc�A�*

loss��9=l��       �	M���Yc�A�*

loss/��;���:       �	C=��Yc�A�*

loss=W��O       �	A׏�Yc�A�*

loss�`�<�_�h       �	�q��Yc�A�*

loss6�R<�҈-       �	���Yc�A�*

lossTg<v��T       �	a���Yc�A�*

lossf�=�xm�       �	�>��Yc�A�*

loss*�<IJ��       �	�璁Yc�A�*

loss��=h�&�       �	o���Yc�A�*

lossQK�<��       �	���Yc�A�*

loss.%=�
��       �	����Yc�A�*

lossC؄=�t�        �	F��Yc�A�*

loss�V�=���       �	'ܕ�Yc�A�*

lossz�!=>�~%       �	V���Yc�A�*

lossC�=B�       �	�(��Yc�A�*

lossړ~=���       �	�͗�Yc�A�*

loss�f�=��I�       �	�k��Yc�A�*

loss��<t��q       �	���Yc�A�*

loss�:};X��\       �	կ��Yc�A�*

loss��=Ave       �	)\��Yc�A�*

loss��<=��8�       �	���Yc�A�*

lossR�<_soG       �	T���Yc�A�*

lossm�<��5�       �	_$��Yc�A�*

loss��-=xY0�       �	b���Yc�A�*

loss4ϝ=�+E       �	�V��Yc�A�*

loss��=�y!%       �	>읁Yc�A�*

loss� �<O���       �	����Yc�A�*

loss�#�=Gt�4       �	���Yc�A�*

loss��=���       �	����Yc�A�*

loss���<҈�       �	J��Yc�A�*

loss8yA<m���       �	?⠁Yc�A�*

loss��<�i�       �	Gw��Yc�A�*

loss�V(<W��W       �	���Yc�A�*

loss<�=��|�       �	'���Yc�A�*

loss��<��Mg       �	*8��Yc�A�*

loss���<��J�       �	�ӣ�Yc�A�*

loss��=����       �	m��Yc�A�*

loss=Nh<���       �	���Yc�A�*

lossqn�<�!p       �	t���Yc�A�*

loss�ah<S�G       �	v2��Yc�A�*

loss��<�e�]       �	 Ʀ�Yc�A�*

loss\==���>       �	�Z��Yc�A�*

loss׺~<��d       �	�Yc�A�*

loss'<�=2��       �	����Yc�A�*

lossw=�=Ϲ?9       �	 D��Yc�A�*

loss�=[�}�       �	ש�Yc�A�*

lossQ�X=u� �       �	�i��Yc�A�*

lossQ�=�SA       �		��Yc�A�*

loss�C�<�b��       �	A���Yc�A�*

loss	��;��_<       �	6>��Yc�A�*

loss{S�<�v       �	[Ҭ�Yc�A�*

lossc��<Ea�       �	{��Yc�A�*

loss��=Ϝ+�       �	���Yc�A�*

loss�6�=�ʶ>       �	Ը��Yc�A�*

loss!nh>D��       �	@Q��Yc�A�*

loss���=ʴ�       �	�证Yc�A�*

lossҿ>BJ�       �	İ�Yc�A�*

loss�"=����       �	1]��Yc�A�*

loss�)=��0�       �	J�Yc�A�*

losst-<����       �	{���Yc�A�*

lossڂY=��-(       �	���Yc�A�*

lossW6=lO��       �	_���Yc�A�*

loss�Ƣ<w4��       �	W��Yc�A�*

loss��=WU`�       �	���Yc�A�*

loss8�Y=���       �	����Yc�A�*

lossO�=��~o       �	�0��Yc�A�*

loss��<,�w�       �	W̶�Yc�A�*

loss-]>��bY       �	3k��Yc�A�*

loss��<�d
E       �	���Yc�A�*

loss3$<�X�>       �	d���Yc�A�*

loss:=	=Tv�%       �	�J��Yc�A�*

loss�p�<p���       �	X乁Yc�A�*

lossƅ=@��       �	,���Yc�A�*

loss��=�6τ       �	m��Yc�A�*

lossz�=J�%�       �	�?��Yc�A�*

lossDa=+��       �	���Yc�A�*

lossn��=�;o�       �	���Yc�A�*

loss��=����       �	�.��Yc�A�*

loss��9=�I�L       �	zȾ�Yc�A�*

loss��<�J�       �	�s��Yc�A�*

lossod=���       �	�`��Yc�A�*

loss �r=N��Q       �	b��Yc�A�*

lossi�R=P��       �	���Yc�A�*

loss��={�,       �	��Yc�A�*

loss��=�p��       �	��ÁYc�A�*

loss*H�<��1       �	�{āYc�A�*

loss��=v�8�       �	�`ŁYc�A�*

lossj)=����       �	ƁYc�A�*

loss�q�=��;(       �	\�ƁYc�A�*

loss�8�=h���       �	�pǁYc�A�*

loss�=q>�       �	�6ȁYc�A�*

lossq<-�1|       �	�ɁYc�A�*

lossv�<{τO       �	X�ɁYc�A�*

loss�>h=�=rs       �	�dʁYc�A�*

loss}�3=k�D       �	!ˁYc�A�*

loss#�'=��       �	��ˁYc�A�*

loss>! =H@~�       �	!=́Yc�A�*

loss3;8=��/�       �	y�́Yc�A�*

loss9�<t��       �	΁Yc�A�*

loss���<�14u       �	/ρYc�A�*

loss�8c=&�N�       �	��ρYc�A�*

loss!˵<�9       �	�~ЁYc�A�*

loss��]=����       �	 �сYc�A�*

lossl�L=�\�?       �	3QҁYc�A�*

loss&��=N���       �	XӁYc�A�*

lossJ��=w���       �	�ԁYc�A�*

loss��>��       �	�ՁYc�A�*

loss�cD=��p*       �	�ՁYc�A�*

loss!��<��ň       �	��ցYc�A�*

lossn �<m�U       �	(�ׁYc�A�*

lossty<�G��       �	�q؁Yc�A�*

loss^�<6#�       �	�(فYc�A�*

losse�>�l��       �	`ځYc�A�*

lossY�=?C��       �	 �ځYc�A�*

loss�ww<4{J       �	�8ہYc�A�*

loss�
<q$�       �	�܁Yc�A�*

lossvY�=���       �	�܁Yc�A�*

loss�z�<�/b       �	�?݁Yc�A�*

loss��<��Z       �	��݁Yc�A�*

loss��<�]�M       �	+�ށYc�A�*

loss��<�\��       �	�%߁Yc�A�*

loss�n�<�e��       �	K�߁Yc�A�*

loss�F=���       �	�d��Yc�A�*

loss�/Y=�$"W       �	��Yc�A�*

loss�d=̓�z       �	��Yc�A�*

loss|@�<��V�       �	 S�Yc�A�*

loss,o=NMj�       �	���Yc�A�*

loss*��;=L�       �	��Yc�A�*

loss�� ;�@nZ       �	�K�Yc�A�*

loss�y�<-��       �	G��Yc�A�*

loss֫<yco�       �	�}�Yc�A�*

loss��n<E��       �	@�Yc�A�*

loss&�<��{�       �	��Yc�A�*

loss�2==��V�       �	"q�Yc�A�*

loss�ɠ<u�C       �	�
�Yc�A�*

loss��A=�<�       �	ߦ�Yc�A�*

lossf�=0A)�       �	SA�Yc�A�*

lossFa1=N(       �	M��Yc�A�*

lossv<=��^�       �	6s�Yc�A�*

lossl�=f�)�       �	"�Yc�A�*

losse/|=!!�Z       �	���Yc�A�*

loss�=m�       �	�A�Yc�A�*

loss{)�=����       �	D��Yc�A�*

loss�0>�T�l       �	�r�Yc�A�*

loss�Hw;�{�       �	�Yc�A�*

loss�XC=q�X       �	n��Yc�A�*

lossR$&=0�j�       �	d<�Yc�A�*

loss.�l=Zf��       �	���Yc�A�*

lossז�<Tu�       �	Mi��Yc�A�*

lossT�V=�j�       �	����Yc�A�*

loss&��=�O��       �	1��Yc�A�*

loss�<�j�8       �	,+�Yc�A�*

lossa`=E&nL       �	G��Yc�A�*

lossG��=U#�       �	�`�Yc�A�*

loss
<R�&       �	���Yc�A�*

loss��|< �|�       �	<��Yc�A�*

lossȊJ=����       �	*��Yc�A�*

loss@i�<t(�O       �	���Yc�A�*

loss ��<�&��       �	lC��Yc�A�*

loss$�2=4}�)       �	+���Yc�A�*

loss�-=�0�N       �	�q��Yc�A�*

loss�	<��       �	���Yc�A�*

loss��=L��       �	����Yc�A�*

lossͤn=x���       �	�3��Yc�A�*

loss��<<�1�       �	q���Yc�A�*

loss��+=2݅       �	�s��Yc�A�*

loss�>=��j�       �	+��Yc�A�*

loss�&>�mX       �	���Yc�A�*

loss��I<)s��       �	h^��Yc�A�*

loss̩<��B       �	���Yc�A�*

loss\�P<�6��       �	����Yc�A�*

loss��<��PS       �	�5��Yc�A�*

lossm�=Txc       �	����Yc�A�*

loss���<�v��       �	�h��Yc�A�*

loss;��=���r       �	�w �Yc�A�*

loss���<2^c       �	
�Yc�A�*

loss�j[<%v��       �	b��Yc�A�*

loss���;�@m�       �	�3�Yc�A�*

loss,��;߁�S       �	q��Yc�A�*

loss�ɑ=d�f�       �	eo�Yc�A�*

loss��u<n���       �	�
�Yc�A�*

loss�lo=!Y��       �	ۣ�Yc�A�*

loss9$>%��       �	a4�Yc�A�*

loss�=-q��       �	���Yc�A�*

lossE�;{�A`       �	�g�Yc�A�*

loss��=YaI       �	&�Yc�A�*

loss�Y=]��       �	ܠ�Yc�A�*

loss�=����       �	�9�Yc�A�*

lossbD<�V��       �	���Yc�A�*

loss=Ad=����       �	z	�Yc�A�*

loss�S�=����       �	�
�Yc�A�*

loss��9=t#a       �	ղ
�Yc�A�*

loss�r�=6��       �	�M�Yc�A�*

loss%?<ciM       �	%��Yc�A�*

lossCI.<$".G       �	��Yc�A�*

loss7��<�+7       �	#�Yc�A�*

loss��M=�榓       �	���Yc�A�*

loss��<}� �       �	���Yc�A�*

losst�7=��W�       �	K#�Yc�A�*

loss���<�o�       �	Ӽ�Yc�A�*

lossv�<�&e�       �	�P�Yc�A�*

loss���=8G�e       �	p��Yc�A�*

lossT�m=ZЏ�       �	'��Yc�A�*

lossQZ=Sťm       �	�9�Yc�A�*

lossA�>�d�I       �	���Yc�A�*

lossBo�=�� �       �	v�Yc�A�*

loss
MC=�ަ[       �	II�Yc�A�*

lossӜ�<��y^       �	���Yc�A�*

loss��=Ѐ�       �	�p�Yc�A�*

loss�y=f>�Y       �	��Yc�A�*

loss	y7=!"�|       �	���Yc�A�*

losss�d;�V       �	g�Yc�A�*

loss�V�=kVk       �	��Yc�A�*

loss��U=��o       �	��Yc�A�*

lossT>�13�       �	F?�Yc�A�*

lossS�b=#Wu
       �	Q��Yc�A�*

loss,A�=O��       �	Tt�Yc�A�*

loss�.�=�Bˢ       �	�G�Yc�A�*

loss��4=�I�+       �	:��Yc�A�*

loss�3=d��       �	���Yc�A�*

loss֌�=P��'       �	)�Yc�A�*

lossR��<,�Q�       �	���Yc�A�*

loss�)
<�й�       �	�f�Yc�A�*

loss�)=��q6       �	 �Yc�A�*

loss`I!=B�S       �	�� �Yc�A�*

loss8NZ=���       �	�K!�Yc�A�*

lossr	=t���       �	��!�Yc�A�*

loss�^=A���       �	T�"�Yc�A�*

loss �j=��^       �	9%#�Yc�A�*

loss��=��l]       �	�#�Yc�A�*

loss��x<�E@�       �	"�$�Yc�A�*

loss�[A<naG       �	$^%�Yc�A�*

lossi94=eFz�       �	�&�Yc�A�*

lossx� >���N       �	�&�Yc�A�*

loss��<nct	       �	h<'�Yc�A�*

loss���<%u3o       �	��'�Yc�A�*

loss]�F<�=       �	��(�Yc�A�*

loss=�<�6HB       �	�)�Yc�A�*

lossOd4<��"`       �	ͱ)�Yc�A�*

loss��2<l�       �	�E*�Yc�A�*

loss2G�=�HJ       �	�-+�Yc�A�*

loss}��<�f4�       �	J,�Yc�A�*

lossIH@=j"��       �	��,�Yc�A�*

loss� e=�i��       �	�|-�Yc�A�*

lossi�=��x       �	B&.�Yc�A�*

loss�(5=�J1t       �	f�.�Yc�A�*

lossT+P=jD�d       �	�V/�Yc�A�*

loss�!=!�:�       �	��/�Yc�A�*

loss��<[A0l       �	ޒ0�Yc�A�*

loss��=��       �	T61�Yc�A�*

lossI��<3�2[       �	��1�Yc�A�*

loss���=ӳ��       �	H�2�Yc�A�*

loss�l�=1w��       �	E3�Yc�A�*

loss{m>�n�       �	
4�Yc�A�*

lossck�;���p       �	T�4�Yc�A�*

lossKB<�\��       �	cD5�Yc�A�*

lossQ�=z���       �	n�5�Yc�A�*

loss�=�>       �	Kw6�Yc�A�*

loss#M�<��e�       �	<7�Yc�A�*

loss'x�=����       �	�7�Yc�A�*

loss,T�<M��       �	�C8�Yc�A�*

lossv��=[~F       �	j�8�Yc�A�*

loss�q=|��       �	�n9�Yc�A�*

loss1|�=�q��       �	�:�Yc�A�*

loss���<�'��       �	�:�Yc�A�*

loss3�s=��^       �	�T;�Yc�A�*

loss� =z?�       �	h�;�Yc�A�*

loss��<�/+T       �	�<�Yc�A�*

loss52=��Wf       �	�)=�Yc�A�*

lossn��<��
/       �	��=�Yc�A�*

loss���<�Km�       �	Yi>�Yc�A�*

loss���<��7       �	�?�Yc�A�*

loss�/@=��       �	��?�Yc�A�*

loss���<5(��       �	l�@�Yc�A�*

loss�A=#��       �	�OA�Yc�A�*

lossqY}<�Io�       �	
�A�Yc�A�*

loss�J<<XO�       �	��B�Yc�A�*

lossّ<O}�       �	��C�Yc�A�*

lossiP�=��\y       �	��D�Yc�A�*

loss�(�=��.       �	NE�Yc�A�*

loss��=Ë��       �	XF�Yc�A�*

lossڷ�=,��       �	�G�Yc�A�*

lossť;Dr�e       �	D�G�Yc�A�*

loss��<߹�K       �	wjH�Yc�A�*

loss�|�=�|e�       �	"I�Yc�A�*

loss���=n�       �	�I�Yc�A�*

lossz=ܢх       �	�SJ�Yc�A�*

loss��=��       �	b�J�Yc�A�*

lossE��=2��       �	6�K�Yc�A�*

loss3õ<�a�v       �	E+L�Yc�A�*

lossw�'<��K       �	p�L�Yc�A�*

lossm�=ͅb�       �	HpM�Yc�A�*

lossT��=��.�       �	�N�Yc�A�*

loss�L�=t8��       �	C�N�Yc�A�*

loss�e=��O�       �	�OO�Yc�A�*

loss�o=@*��       �	|�O�Yc�A�*

lossL��<�\ x       �	&�P�Yc�A�*

lossab�=>�       �	TQ�Yc�A�*

lossqў=��       �	��Q�Yc�A�*

loss�
�<���       �	�NR�Yc�A�*

lossv�<�aL       �	��R�Yc�A�*

loss6�E=��_       �	�S�Yc�A�*

loss�{<�?Y�       �	rQT�Yc�A�*

loss�x�<���       �	��T�Yc�A�*

loss���<wC
�       �	F�U�Yc�A�*

loss�v�<���&       �	_V�Yc�A�*

lossx�I=+D��       �	�	W�Yc�A�*

loss-_o<�y��       �	��W�Yc�A�*

lossD�=�&q       �	_\X�Yc�A�*

lossQ�P=i� s       �	\ Y�Yc�A�*

lossn7�=:���       �	Y�Y�Yc�A�*

loss`��=TrG       �	�KZ�Yc�A�*

lossn(�<�ˬV       �	R�Z�Yc�A�*

loss�+�;J
ă       �	�[�Yc�A�*

lossZnP=0��       �	�:\�Yc�A�*

lossS�<�lmW       �	��\�Yc�A�*

loss�=��l       �	;�]�Yc�A�*

loss=��=��Q�       �	�,^�Yc�A�*

loss�P�=�?�       �	�^�Yc�A�*

loss�OP=C��       �	�v_�Yc�A�*

loss�*=�L�       �	�`�Yc�A�*

loss��=��&h       �	��`�Yc�A�*

loss�=�?�       �	Xpa�Yc�A�*

loss@n�<RT��       �	�b�Yc�A�*

loss��<4Ñ�       �	ɮb�Yc�A�*

loss�ہ=��       �	�Qc�Yc�A�*

loss82�<�\�       �	��c�Yc�A�*

loss֨�<%��       �	��d�Yc�A�*

loss�֖<b���       �	��e�Yc�A�*

loss$Շ=�Xu
       �	9%f�Yc�A�*

lossh��;G�<�       �	}g�Yc�A�*

loss�_<)9�I       �	b�g�Yc�A�*

loss�|<je�       �	%>h�Yc�A�*

loss�-�<��H�       �	Q�h�Yc�A�*

loss�ۊ=�k��       �	�oi�Yc�A�*

loss�&�;�<�       �	�j�Yc�A�*

loss��<tõ       �	�j�Yc�A�*

loss�.==��H)       �	Zk�Yc�A�*

loss��p=�ժ       �	�l�Yc�A�*

loss3se<��m�       �	i�l�Yc�A�*

loss#/�<�K�"       �	*Um�Yc�A�*

loss�+7<���       �	j�m�Yc�A�*

loss�F�=�B��       �	4�n�Yc�A�*

lossE��<��8       �	Ro�Yc�A�*

loss���=�7�       �	��o�Yc�A�*

loss��<&9b       �	�p�Yc�A�*

loss:S>S��\       �	�>q�Yc�A�*

loss<�=⮁�       �	`�q�Yc�A�*

loss ��<|��       �	șr�Yc�A�*

loss�x�=�2�       �	�Is�Yc�A�*

loss.��<=	��       �	�s�Yc�A�*

loss�l>>�       �	��t�Yc�A�*

loss�S<�<��       �	�Vu�Yc�A�*

loss!�j=�vH�       �	�v�Yc�A�*

loss���<���       �	��v�Yc�A�*

loss[-O=|xy�       �	Ƈw�Yc�A�*

loss�v�<�`v~       �	�5x�Yc�A�*

loss�)=G-4�       �	x�x�Yc�A�*

loss�<�e7       �	�y�Yc�A�*

loss�=�@r       �	RDz�Yc�A�*

loss�;Ϛ�d       �	��z�Yc�A�*

loss��D<�/�8       �	��{�Yc�A�*

lossx6�<�o�~       �	O|�Yc�A�*

loss�H�<Y� �       �	Y�|�Yc�A�*

losszP�<�Ҽ�       �	?�}�Yc�A�*

lossI�>���/       �	mT~�Yc�A�*

losssD=�"��       �	@0�Yc�A�*

loss:�<�n�i       �	4��Yc�A�*

loss)4�;D��       �	4���Yc�A�*

lossS�T=��       �	Y1��Yc�A�*

loss�2!<� .6       �	���Yc�A�*

loss�&;� ��       �	I��Yc�A�*

loss('�;� �J       �	~ă�Yc�A�*

loss�V�;�u��       �	s�Yc�A�*

lossd/l<�1�L       �	���Yc�A�*

lossR��;��       �	����Yc�A�*

loss2<k���       �	�r��Yc�A�*

lossŜ|<�n       �	�]��Yc�A�*

lossI�;����       �	�r��Yc�A�*

lossDgk:�ݕ       �	�;��Yc�A�*

loss��m:���       �	[��Yc�A�*

loss�)<���       �	&ዂYc�A�*

loss3$=���(       �	���Yc�A�*

loss!��<��R       �	=��Yc�A�*

lossMT$:7o̸       �	��Yc�A�*

loss�)=8��(       �	����Yc�A�*

loss�	>���       �	Ve��Yc�A�*

loss ��:{��P       �	�'��Yc�A�*

lossCD>����       �	Gʐ�Yc�A�*

loss~1�=/� �       �	?q��Yc�A�*

loss
Û=�L�       �	��Yc�A�*

lossh��<��C�       �	�Œ�Yc�A�*

loss/�$=�<�       �	 p��Yc�A�*

loss���=���       �	���Yc�A�*

loss
��=KM�q       �	��Yc�A�*

loss;2^=���       �	zn��Yc�A�*

lossa�I=Uu       �	���Yc�A�*

loss�wz=�'�       �	ᶖ�Yc�A�*

loss:>�f�       �	[\��Yc�A�*

loss��>N�?�       �	����Yc�A�*

loss�U�<�I��       �	����Yc�A�*

loss�1v=�S\#       �	�+��Yc�A�*

loss�+Y=vSey       �	ș�Yc�A�*

loss�R�<�W�       �	�x��Yc�A�*

lossN%=\
}	       �	���Yc�A�*

loss�>���       �	����Yc�A�*

loss�=�O�       �	�O��Yc�A�*

loss��
<�&�       �	K蜂Yc�A�*

loss؈=έ�-       �	e���Yc�A�*

loss<�^=�i�v       �	%#��Yc�A�*

loss���<��U�       �	����Yc�A�*

loss/;�<�x�       �	�\��Yc�A�*

loss,�<���       �	v���Yc�A�*

loss���<k��G       �	h���Yc�A�*

loss��=*�v       �	�-��Yc�A�*

loss-� >�	��       �	ɡ�Yc�A�*

lossX�=/'�       �	�s��Yc�A�*

lossE1�<�%�       �	+��Yc�A�*

lossSO=�	:3       �	����Yc�A�*

loss.�c=p�D7       �	�T��Yc�A�*

lossJ=<�Od       �	& ��Yc�A�*

loss��<sR��       �	�p��Yc�A�*

loss��<�O�       �	���Yc�A�*

lossW|;<5>�`       �	qȧ�Yc�A�*

lossi��=��	m       �	h^��Yc�A�*

loss�ԏ=�^cc       �	����Yc�A�*

loss��a=�(�i       �	󑩂Yc�A�*

loss=K�<��]       �	3��Yc�A�*

loss�<7\)�       �	�Ǫ�Yc�A�*

loss��N=3�t�       �	rl��Yc�A�*

loss�K=[���       �	���Yc�A�*

loss��<�u;z       �	�٬�Yc�A�*

lossM~1=��v�       �	Ot��Yc�A�*

losskz=���       �	�"��Yc�A�*

loss]�<�8��       �	����Yc�A�*

loss(;=Ğ9K       �	�a��Yc�A�*

lossZ�<���       �	����Yc�A�*

lossh�<lX�G       �	˜��Yc�A�*

loss���=x�D       �	��͂Yc�A�*

loss��=M>�       �	)Y΂Yc�A�*

lossJ;>��       �	��΂Yc�A�*

loss��`=�n�       �	q�ςYc�A�*

lossU�"=P��s       �	�$ЂYc�A�*

lossE@�<�[�       �	k�ЂYc�A�*

loss��w<��J       �	eSтYc�A�*

loss���<��	       �	!҂Yc�A�*

loss���=���       �	A�҂Yc�A�*

lossQ?�=�       �	�cӂYc�A�*

loss%R�<=��       �	��ӂYc�A�*

lossr=K=��F"       �	��ԂYc�A�*

loss��=Osǵ       �	a4ՂYc�A�*

loss[/=��9$       �	��ՂYc�A�*

loss2
�<w���       �	k~ւYc�A�*

loss���<��[%       �	W!ׂYc�A�*

loss �;!��_       �	��ׂYc�A�*

loss�j0=��       �	:u؂Yc�A�*

loss�<�b       �	TڂYc�A�*

lossz�_=�NZ�       �	d�ڂYc�A�*

lossx>�<��,       �	�IۂYc�A�*

loss)��=|�j       �	�ۂYc�A�*

loss<Ow<N�RZ       �	u�܂Yc�A�*

loss�=�聚       �	�0݂Yc�A�*

loss��<;�,r       �	��݂Yc�A�*

loss�#�<{E<�       �	M�ނYc�A�*

loss�<8=��7�       �	�-߂Yc�A�*

losso��;(���       �	�߂Yc�A�*

loss�ϖ<���       �	�e��Yc�A�*

loss�=<��]T       �	�Yc�A�*

lossE
�<��       �	��Yc�A�*

lossQ�@<�xV       �	OY�Yc�A�*

loss�0=��       �	w��Yc�A�*

loss��<��b�       �	���Yc�A�*

loss��k='��c       �	^-�Yc�A�*

loss�hE=��       �	���Yc�A�*

loss�x�;��k       �	�f�Yc�A�*

lossd�<��U�       �	�e�Yc�A�*

lossa�=mbY<       �	�i�Yc�A�*

lossO�=��&�       �	��Yc�A�*

lossv<=���6       �	���Yc�A�*

loss&b=��ޜ       �	�h�Yc�A�*

lossD�<xIj       �	��Yc�A�*

loss!_\=�@�       �	4��Yc�A�*

loss&�<�;��       �	d:�Yc�A�*

loss�F�=�ڦS       �	J��Yc�A�*

lossR�+=�       �	�p�Yc�A�*

lossl��<}�.       �	��Yc�A�*

lossq@�;��*       �	<��Yc�A�*

lossv0;gq��       �	�z�Yc�A�*

loss��<�/X       �	�5�Yc�A�*

loss��<LIap       �	���Yc�A�*

lossc�=����       �	�{��Yc�A�*

loss.��=8iK       �	O �Yc�A�*

loss�|�<��       �	���Yc�A�*

loss�Jh;k�       �	d]�Yc�A�*

loss]�;�n�       �	  �Yc�A�*

loss�[<\���       �	f��Yc�A�*

loss�MK=HV[       �	cB�Yc�A�*

loss��Z<��(5       �	���Yc�A�*

lossl��=hRX       �	�}��Yc�A�*

loss�.'=�wb�       �	E)��Yc�A�*

losst��<� �       �	����Yc�A�*

loss���<s|&       �	�e��Yc�A�*

lossi'�;���j       �	i���Yc�A�*

loss�=
y�       �	*���Yc�A�*

loss2� ==��       �	@O��Yc�A�*

loss�,=ư�       �	p���Yc�A�*

loss�"=�խ&       �	���Yc�A�*

loss$��=�)m�       �	�+��Yc�A�*

loss��I=:�Ħ       �	 ���Yc�A�*

loss�t	=�1��       �	[^��Yc�A�*

loss�ً<S�(x       �	���Yc�A�*

loss�� =tR�       �	4���Yc�A�*

loss�=�<�X�       �	:��Yc�A�*

loss&��<kU�       �	����Yc�A�*

loss�i=��;�       �	;o��Yc�A�*

loss�!�=+���       �	zn �Yc�A�*

loss�*�<����       �	A�Yc�A�*

loss�5=5���       �	���Yc�A�*

loss`ˣ=��       �	O�Yc�A�*

loss�!9=�Ƕ�       �	$��Yc�A�*

loss�k,=��|{       �	���Yc�A�*

loss��<>̮�       �	!<�Yc�A�*

lossx4=��;�       �	���Yc�A�*

loss���<��m�       �	fk�Yc�A�*

loss/P;�vA�       �	��Yc�A�*

lossI�<�"��       �	���Yc�A�*

loss�^s<��V�       �	�<�Yc�A�*

loss
j=�a^g       �	*��Yc�A�*

loss@Ű<��s       �	x�Yc�A�*

lossWۧ=�&t/       �	b	�Yc�A�*

lossf,.=�j       �	��	�Yc�A�*

lossS�=Ffn�       �	��
�Yc�A�*

lossܭ<?�jt       �	�I�Yc�A�*

loss���<����       �	���Yc�A�*

lossn�
>�F�       �	�{�Yc�A�*

loss�9=�A~       �	��Yc�A�*

loss=�B=��G       �	ɭ�Yc�A�*

lossϚx<��       �	[D�Yc�A�*

loss�=��A�       �	j��Yc�A�*

loss�<���2       �	0��Yc�A�*

loss�2X<���e       �	�$�Yc�A�*

loss+@<e�F       �	w��Yc�A�*

loss�0�=���       �	�W�Yc�A�*

loss�/�=e�'�       �	 �Yc�A�*

loss[�w; �q
       �	'��Yc�A�*

loss�<Ev�       �	B�Yc�A�*

loss���=��       �	���Yc�A�*

loss�
<�|�       �	���Yc�A�*

loss��=���y       �	�a�Yc�A�*

lossH��=jn�^       �	m��Yc�A�*

lossq&�<n�j�       �	Ϡ�Yc�A�*

lossvR;�LO�       �	�C�Yc�A�*

loss�Xi<�z�       �	���Yc�A�*

loss_�B<N}�       �	�{�Yc�A�*

loss02
=0��       �	��Yc�A�*

loss&�="��P       �	��Yc�A�*

loss��=���       �	$��Yc�A�*

loss�O�;�꾠       �	fN�Yc�A�*

loss�<d�5       �	Z+�Yc�A�*

lossF�\=�/��       �	d��Yc�A�*

loss�y�<gR�       �	�t�Yc�A�*

loss<��<�?2       �	��Yc�A�*

lossA�;��1`       �	���Yc�A�*

lossK��=���1       �	S[�Yc�A�*

lossa�E=��>       �	� �Yc�A�*

loss�?�<���_       �	� �Yc�A�*

lossOƲ=G�C,       �	�L!�Yc�A�*

loss�4�<����       �	,�!�Yc�A�*

loss�3=��       �	��"�Yc�A�*

lossj�=e��       �	f#�Yc�A�*

lossɎz=�9       �	�#�Yc�A�*

loss���<�l�       �	��$�Yc�A�*

loss��=}zi�       �	�>%�Yc�A�*

loss�Ey< ��O       �	��%�Yc�A�*

lossh$=�T!�       �	x�&�Yc�A�*

lossQ�!=e~ӎ       �	��'�Yc�A�*

loss{��<����       �	��(�Yc�A�*

loss�&U<�}U       �	/)�Yc�A�*

loss|q�;`��       �	�*�Yc�A�*

loss��=E�       �	��*�Yc�A�*

lossme=8       �	v�+�Yc�A�*

lossj��<�è�       �	�q,�Yc�A�*

loss�=���m       �	-�Yc�A�*

loss��!=��       �	��-�Yc�A�*

loss��<?���       �	��.�Yc�A�*

loss<2�       �	5~/�Yc�A�*

loss��=�uT       �	#k0�Yc�A�*

loss4|*<�pS�       �	1(1�Yc�A�*

loss��<~�       �	U�1�Yc�A�*

loss��(=���3       �	�p2�Yc�A�*

lossq��<x�       �	�3�Yc�A�*

loss/5v<��N�       �	�3�Yc�A�*

loss%��=���       �	�G4�Yc�A�*

loss�<�AW�       �	��4�Yc�A�*

loss��3<�%Hu       �	?s5�Yc�A�*

loss��F=�Y��       �	�6�Yc�A�*

loss�@=ɂ��       �	{�6�Yc�A�*

loss;q.<�u�       �	�=7�Yc�A�*

loss��=��       �	��7�Yc�A�*

loss.��<��&�       �	�l8�Yc�A�*

loss�=���[       �	�9�Yc�A�*

lossa;`.��       �	��9�Yc�A�*

loss"W=S-�9       �	n:�Yc�A�*

lossVT�<� �7       �	;�Yc�A�*

loss���<�q3       �	-�;�Yc�A�*

loss���=��L       �	Y3<�Yc�A�*

lossʽ�<�`�~       �	��<�Yc�A�*

lossq0�<�ŋ       �	�b=�Yc�A�*

loss��7<�{��       �	��=�Yc�A�*

loss��;��R�       �	y�>�Yc�A�*

loss�d?<mn�M       �	^/?�Yc�A�*

loss�[>�F�       �	X�?�Yc�A�*

loss�!5=߷�X       �	�d@�Yc�A�*

loss8�=���       �	�!A�Yc�A�*

lossRi�<����       �	۾A�Yc�A�*

loss(R=�=t�       �	��B�Yc�A�*

loss���=v���       �	�dC�Yc�A�*

lossW�<Y�       �	}D�Yc�A�*

loss�f�;U[�5       �	A�D�Yc�A�*

loss[a=�jR�       �	+�E�Yc�A�*

loss�ɵ<a�       �	�mF�Yc�A�*

loss�3�;��g�       �	�	G�Yc�A�*

lossI[�;�b�T       �	YH�Yc�A�*

lossؠ;�+G       �	ɮH�Yc�A�*

loss�l==|n��       �	`VI�Yc�A�*

loss�.�<l#~�       �	��I�Yc�A�*

loss��=H�A       �	l�J�Yc�A�*

lossTQ?=jS�       �	36K�Yc�A�*

loss� >ƪ�z       �	��K�Yc�A�*

loss/�=)C��       �	�iL�Yc�A�*

loss%�-<��A�       �	dN�Yc�A�*

lossd�q<ag�K       �	�N�Yc�A�*

loss�M;B��       �	q:O�Yc�A�*

loss�6_=����       �	�P�Yc�A�*

loss���<K O�       �	�P�Yc�A�*

loss���=6V�s       �	sJQ�Yc�A�*

loss��=Z�F�       �	��Q�Yc�A�*

loss�!<�̅u       �	&�R�Yc�A�*

loss�3�=-���       �	xFS�Yc�A�*

loss<P<�~8       �	f�S�Yc�A�*

loss��;#8��       �	ؚT�Yc�A�*

loss��=�$�       �	G>U�Yc�A�*

lossU<,�?O       �	��U�Yc�A�*

lossG�=B�       �	��V�Yc�A�*

loss�)x=����       �	�oW�Yc�A�*

losse= V��       �	�X�Yc�A�*

lossx�=G�X~       �	W�X�Yc�A�*

loss6<n�|       �	�MY�Yc�A�*

loss��;<��H>       �	.�Y�Yc�A�*

loss��=q�       �	�Z�Yc�A�*

loss=�&=;:m5       �	�.[�Yc�A�*

lossd�=�*K       �	��[�Yc�A�*

loss�0=����       �	Ow\�Yc�A�*

loss��=ʿ�       �	O]�Yc�A�*

loss�z�=.^�       �	��]�Yc�A�*

loss�s=U�QO       �	�V^�Yc�A�*

loss���=jU;        �	(�^�Yc�A�*

loss
�<����       �	M�_�Yc�A�*

loss�<�Ve�       �	�B`�Yc�A�*

loss�\=-��       �	��`�Yc�A�*

loss� =m�       �	�a�Yc�A�*

loss�<S=Y_��       �	"5b�Yc�A�*

lossdH<�[4�       �	��b�Yc�A�*

lossO��<��NU       �	I�c�Yc�A�*

loss�(=�~�       �	�d�Yc�A�*

loss
��<3��       �	q�d�Yc�A�*

lossj=�d��       �	�me�Yc�A�*

loss;C$=�:�_       �	�)f�Yc�A�*

loss�&=��m�       �	��f�Yc�A�*

loss@Y<(&U�       �	�`g�Yc�A�*

loss%6�;�S�       �	�g�Yc�A�*

loss/A�<L��       �	p�h�Yc�A�*

loss���=M+P�       �	W�i�Yc�A�*

losst�z=tO��       �	�ij�Yc�A�*

loss��W=`�dG       �	�k�Yc�A�*

loss�@�=.m��       �	)�k�Yc�A�*

lossn��=��j`       �	�Kl�Yc�A�*

loss�!e<כt�       �	�wm�Yc�A�*

loss�ˇ<�;p       �	�en�Yc�A�*

lossNa=�"��       �	No�Yc�A�*

loss��^=�K�       �	��o�Yc�A�*

loss� �<M��       �	Egp�Yc�A�*

loss�,|=&8R$       �	qq�Yc�A�*

lossO<ǯ
�       �	'�q�Yc�A�*

loss���<o|       �	_�r�Yc�A�*

lossX�=�=F�       �	8/s�Yc�A�*

loss7=c��       �	%�s�Yc�A�*

loss <�<1	       �	��t�Yc�A�*

loss�^=���       �	(cu�Yc�A�*

loss6J�=3���       �	/�u�Yc�A�*

lossd2�<m�3l       �	��v�Yc�A�*

loss�$�;vt-p       �	>]w�Yc�A�*

loss_#;Z/��       �	��w�Yc�A�*

lossL<�ш�       �	,�x�Yc�A�*

loss�=��y       �	�Cy�Yc�A�*

loss�=�eL�       �	��y�Yc�A�*

loss��<��KT       �	�z�Yc�A�*

loss�Ɇ=��<f       �	+{�Yc�A�*

lossRdn=��S       �	�{�Yc�A�*

lossl��=��1       �	�[|�Yc�A�*

lossl��<zͩD       �	��|�Yc�A�*

loss��=�}��       �	i�}�Yc�A�*

loss���=�       �	c~�Yc�A�*

loss��"<�=4u       �	��~�Yc�A�*

lossO�
=>�9�       �	��Yc�A�*

loss7�F=g��       �	Iڀ�Yc�A�*

loss���=%=�       �	���Yc�A�*

loss9<=p�vI       �	�A��Yc�A�*

loss�?h=�qE       �	�݂�Yc�A�*

loss�Ç=g���       �	����Yc�A�*

loss	>�<�7��       �	r��Yc�A�*

loss�,�<qx\       �	�:��Yc�A�*

loss]�?>"�'l       �		ޅ�Yc�A�*

loss��>]�@       �	舆�Yc�A�*

loss��J=�}       �	�n��Yc�A�*

loss+=`�#�       �	���Yc�A�*

loss6P=�5�       �	���Yc�A�*

loss�<X[�K       �	�7��Yc�A�*

loss_=<�AX       �	�Yc�A�*

loss�;�=�g�Y       �	����Yc�A�*

loss��&<�ju       �	u9��Yc�A�*

loss���=���       �	J팃Yc�A�*

loss6x@=�"��       �	Ɗ��Yc�A�*

loss�N�<%i��       �	�;��Yc�A�*

loss�m�=�6�       �	�܎�Yc�A�*

lossI��;��l       �	m���Yc�A�*

lossj&<��6       �	Q-��Yc�A�*

lossW�;��'       �	�ΐ�Yc�A�*

loss�A;!��       �	{��Yc�A�*

loss<�T=S�\_       �	>#��Yc�A�*

loss�h�;v�>Q       �	1ђ�Yc�A�*

loss�rs=�O:s       �	3���Yc�A�*

loss��N<�;��       �	'.��Yc�A�*

loss���=P�       �	˔�Yc�A�*

loss�z�<��       �	r��Yc�A�*

loss�M�=�Ԁ�       �	N��Yc�A�*

loss]�>�}-       �	����Yc�A�*

loss��1=� ��       �	�D��Yc�A�*

loss+�<�Ha�       �	8ܗ�Yc�A�*

loss��=�[�       �	S���Yc�A�*

loss�X=��ض       �	J��Yc�A�*

loss@��<u�       �	�虃Yc�A�*

loss	.�=��       �	
���Yc�A�*

loss��>��W�       �	� ��Yc�A�*

loss�uf<��T       �	����Yc�A�*

lossEY�=�P�       �	aS��Yc�A�*

loss�w<�dB&       �	Yc�A�*

loss�K#<��	       �	����Yc�A�*

loss%M�<���       �	�9��Yc�A�*

loss{T7=�ε       �	G垃Yc�A�*

loss�£=kǩ�       �	5~��Yc�A�*

loss�x�<��^�       �	�'��Yc�A�*

loss�>�<���7       �	ۿ��Yc�A�*

lossK�<�.}F       �	�Y��Yc�A�*

lossÝ�<�G}�       �	���Yc�A�*

loss�%=�kF       �	U���Yc�A�*

lossd�=|,��       �	�?��Yc�A�*

loss ]2=���	       �	�٣�Yc�A�*

loss kU=��       �	����Yc�A�*

loss�+=���5       �	�&��Yc�A�*

loss��<��"d       �	�ߥ�Yc�A�*

loss���<@��       �	y��Yc�A�*

loss���<�Z��       �	C��Yc�A�*

loss�`n=���       �	F���Yc�A�*

lossRh-<Cl0        �	�P��Yc�A�*

loss��==&+�       �	��Yc�A�*

loss�rD<�V��       �	擩�Yc�A�*

lossm�=od.       �	^-��Yc�A�*

lossa�`<��Z�       �	�Ǫ�Yc�A�*

loss\ص<h�{       �	�c��Yc�A�*

losst�<���       �	�
��Yc�A�*

lossn��;���       �	o���Yc�A�*

loss#��<i8�       �	�e��Yc�A�*

loss=�M=�5E�       �	��Yc�A�*

loss$�=A�\       �	����Yc�A�*

loss;�<7<�       �	�O��Yc�A�*

loss_\�=i>       �	n���Yc�A�*

loss�<]#�       �	ܛ��Yc�A�*

loss��=���       �	�9��Yc�A�*

loss��6<�,~�       �	4ڱ�Yc�A�*

loss��)=�<@       �	���Yc�A�*

loss��O<�i�-       �	�(��Yc�A�*

loss�b>���b       �	ӳ�Yc�A�*

loss��y=��       �	Pp��Yc�A�*

losst�X<�Є�       �	�!��Yc�A�*

loss /�=�fC�       �	����Yc�A�*

lossmB<=�^
�       �	�o��Yc�A�*

loss: �=g�       �	���Yc�A�*

loss<��<�DBG       �	f���Yc�A�*

loss
J�<CN�1       �	:\��Yc�A�*

loss =�I       �	Z��Yc�A�*

loss�� =E��w       �	Ϻ��Yc�A�*

loss�'=mx&l       �	�a��Yc�A�*

loss��;Cc�O       �	[��Yc�A�*

loss�<i�˅       �	g���Yc�A�*

loss�K�<R�j       �	�c��Yc�A�*

loss:&f<|�       �	o��Yc�A�*

loss��&<1�"�       �	����Yc�A�*

loss��<=��{	       �	i��Yc�A�*

loss�=�E�`       �	���Yc�A�*

loss��&=>}        �	����Yc�A�*

loss!��=Ŀ��       �	���Yc�A�*

loss��=�cFD       �	>>��Yc�A�*

loss���=����       �	;Yc�A�*

loss���=�"�c       �	�Yc�A�*

loss%3�<iW�       �	��ÃYc�A�*

loss���;GXW       �	l�ăYc�A�*

lossw�=G��       �	g�ŃYc�A�*

loss�?�<z��       �	P�ƃYc�A�*

loss$<'�	       �	Y�ǃYc�A�*

lossIUj=�~�       �	?ȃYc�A�*

loss��s=��       �	��ȃYc�A�*

loss8.=R7U�       �	΋ɃYc�A�*

loss{H�=��w�       �	~:ʃYc�A�*

loss�rO=�B��       �	��ʃYc�A�*

loss�a=4Nm�       �	Ō˃Yc�A�*

lossA�T<��1
       �	f2̃Yc�A�*

loss\b�=m��V       �	��̃Yc�A�*

loss+��<%y��       �	�̓Yc�A�*

lossvv�<#�q       �	�^΃Yc�A�*

loss.p�=r��]       �	�σYc�A�*

loss��u=x'       �	�σYc�A�*

loss/��;sZ��       �	uЃYc�A�*

loss�l�=�қ{       �	�уYc�A�*

loss25�<_�U       �	A�уYc�A�*

loss2�T=o�،       �	,b҃Yc�A�*

lossL~�<���       �	~ӃYc�A�*

loss��<���       �	�ӃYc�A�*

loss���<C���       �	^MԃYc�A�*

loss�k=��e       �	��ԃYc�A�*

loss�w�<ہ̇       �	X�ՃYc�A�*

loss.�F<n       �	pBփYc�A�*

loss�=ZJ}       �	1�փYc�A�*

loss8�=f�{       �	��׃Yc�A�*

losso,<]�j       �	�-؃Yc�A�*

lossnH�<����       �	��؃Yc�A�*

loss��<���%       �	LoكYc�A�*

lossѷ�=㏒�       �	{ڃYc�A�*

loss�d.<g���       �	�ڃYc�A�*

loss��W<����       �	�^ۃYc�A�*

lossq�=�J�]       �	�܃Yc�A�*

losse�;$8�       �	�܃Yc�A�*

loss�_"=Թ.�       �	rS݃Yc�A�*

loss�N�<��^,       �	��݃Yc�A�*

loss��=�S�A       �	d�ރYc�A�*

losss�<��$       �	y;߃Yc�A�*

loss���<�3�       �	n�߃Yc�A�*

lossߨi=�k�	       �	����Yc�A�*

loss|5�<�Vjw       �	9�Yc�A�*

lossT~=SU�       �	���Yc�A�*

loss&1�<����       �	!��Yc�A�*

loss!�R=z"C4       �	�6�Yc�A�*

loss�C@=r��       �	W��Yc�A�*

loss�8�=(m�#       �	̘�Yc�A�*

lossHM�;�]j�       �	oK�Yc�A�*

loss�f2<pw��       �	V��Yc�A�*

loss�I�<tKx>       �	��Yc�A�*

lossS��=x���       �	�>�Yc�A�*

loss��<{L��       �	���Yc�A�*

loss� �=]P��       �	k��Yc�A�*

loss�`�=���       �	��Yc�A�*

loss���=[}�g       �	���Yc�A�*

loss��=��x       �	>[�Yc�A�*

lossī@<�{�>       �	���Yc�A�*

losstr-<��s       �	���Yc�A�*

loss�E;=�>�Z       �	�G�Yc�A�*

lossw�<����       �	���Yc�A�*

loss`9M=Ojv�       �	�}�Yc�A�*

loss&�w<���T       �	-%�Yc�A�*

loss��=x�R       �	���Yc�A�*

loss*&�<���       �	�X�Yc�A�*

loss��=�b�p       �	���Yc�A�*

loss�a=�!�       �	���Yc�A�*

loss]�<����       �	P9�Yc�A�*

lossќ<���7       �	[��Yc�A�*

loss��W=��/�       �	�n�Yc�A�*

loss��r<�Dw       �	�Yc�A�*

lossԸ'<4E�B       �	ɫ�Yc�A�*

loss�=l0�%       �	�I�Yc�A�*

lossB>p>/�       �	U��Yc�A�*

lossl �=%�=       �	){��Yc�A�*

loss�@X=���q       �	s��Yc�A�*

loss.��;�ā       �	&���Yc�A�*

loss���=��!�       �	�?��Yc�A�*

loss�J_=���$       �	����Yc�A�*

losst��;�&Q       �	�s��Yc�A�*

loss� �<I,;       �	���Yc�A�*

loss���=�Ty       �	U���Yc�A�*

loss�)_=�sՓ       �	tC��Yc�A�*

lossO�!<H��       �	����Yc�A�*

lossqT�;>��       �	Ts��Yc�A�*

loss��w=Ig�3       �	�	��Yc�A�*

lossO�H=)�yK       �	I���Yc�A�*

lossv��=/q�{       �	�3��Yc�A�*

loss3p=��߄       �	����Yc�A�*

loss2r<�Z�       �	�s��Yc�A�*

loss�W�<��	       �	���Yc�A�*

loss1HG<�\�A       �	@���Yc�A�*

loss =[<��Py       �	`< �Yc�A�*

loss��<A�:       �	� �Yc�A�*

loss��6;��n�       �	��Yc�A�*

loss7��=OE��       �	CU�Yc�A�*

lossܨ3<&ؖ�       �	q�Yc�A�*

loss<Z�G       �	���Yc�A�*

loss)��<�V��       �	x��Yc�A�*

loss#5*=�x`:       �	C��Yc�A�*

loss�!z<(N\�       �	m��Yc�A�*

loss�<���i       �	��Yc�A�*

loss���=𷻟       �	�_�Yc�A�*

lossɉ=��U�       �	�"	�Yc�A�*

loss���<ۻT)       �	
�	�Yc�A�*

loss�_=��       �	��
�Yc�A�*

loss���<�t|�       �	8��Yc�A�*

loss%�_<n#�       �	%t�Yc�A�*

loss� <5W�u       �	]S�Yc�A�*

loss�0�=}� �       �	[_�Yc�A�*

loss�<穞       �	Z-�Yc�A�*

loss_CK=A�2E       �	=+�Yc�A�*

loss� �=��$c       �	e��Yc�A�*

loss�33=U�2�       �	 `�Yc�A�*

loss�;_*H�       �	��Yc�A�*

loss��=��;�       �	���Yc�A�*

losse_=d/�       �	�.�Yc�A�*

loss1��<�gaR       �	9��Yc�A�*

loss�~�<�p        �	nn�Yc�A�*

losss�=o6>       �	%!�Yc�A�*

loss�%<� v       �	���Yc�A�*

loss�=-R96       �	]�Yc�A�*

loss��<���       �	.��Yc�A�*

loss�|�=�U��       �	���Yc�A�*

loss!	<~#��       �	I0�Yc�A�*

loss%�=�R��       �	���Yc�A�*

lossl><�@�R       �	aq�Yc�A�*

lossA�n<ncbA       �	�Yc�A�*

loss�H�=>�Y       �	���Yc�A�*

loss�9�;��       �	�@�Yc�A�*

loss}}J=���P       �	���Yc�A�*

loss:=���       �	�{�Yc�A�*

loss�U=�       �	��Yc�A�*

loss� �<�a�       �	���Yc�A�*

loss�G<��I       �	�J�Yc�A�*

loss�b$<���       �	���Yc�A�*

loss���=���       �	A~�Yc�A�*

loss��=����       �	M �Yc�A�*

lossH�^=G�M�       �	i� �Yc�A�*

loss�F�<�̔       �	I!�Yc�A�*

loss�F=�(��       �	�!�Yc�A�*

loss�1#=&�\K       �	^�"�Yc�A�*

loss���<'�	       �	#�Yc�A�*

loss�!�=�)       �	ݲ#�Yc�A�*

lossT^�<tMB       �	(I$�Yc�A�*

lossA@�=���       �	T�$�Yc�A�*

lossA� =���&       �	�v%�Yc�A�*

loss�c�=r��       �	�&�Yc�A�*

lossT�t=��bq       �	P�&�Yc�A�*

lossӢ�<tLSI       �	�<'�Yc�A�*

lossm�q;z���       �	�(�Yc�A�*

loss��=��r5       �	��(�Yc�A�*

loss�u&<�D��       �	�S)�Yc�A�*

loss�m�<��R�       �	��)�Yc�A�*

lossR��;E\�       �	��*�Yc�A�*

loss��=�z6       �	�,+�Yc�A�*

loss ��<1垿       �	��+�Yc�A�*

loss�~2=���       �	�w,�Yc�A�*

loss�G�<X��       �	{-�Yc�A�*

loss�j�=���f       �	$�-�Yc�A�*

lossi�;=4 h�       �	i.�Yc�A�*

loss��"<Q<ʄ       �	�/�Yc�A�*

lossp�<��2x       �	�0�Yc�A�*

loss=(�=o*�C       �	д0�Yc�A�*

loss�/K<m�?
       �	ML1�Yc�A�*

loss/�j<���       �	�1�Yc�A�*

loss�;|� 2       �	�~2�Yc�A�*

loss��A<GX`�       �	�<3�Yc�A�*

lossJ6%<q!�       �	��3�Yc�A�*

loss�n[;o!�       �	�n4�Yc�A�*

loss�<�<G���       �	�5�Yc�A�*

loss'�<��)g       �	�5�Yc�A�*

loss�^R;��D�       �	a56�Yc�A�*

lossd`:���A       �	%�6�Yc�A�*

loss�;�:V4E�       �	�m7�Yc�A�*

loss��:<T_$Y       �	8�Yc�A�*

loss�<�4��       �	u�8�Yc�A�*

lossl��;���       �	�|9�Yc�A�*

loss�]�:z�       �	
:�Yc�A�*

loss�o=� }�       �	�:�Yc�A�*

lossEbJ>��       �	�d;�Yc�A�*

loss�y<;�w�       �	��;�Yc�A�*

loss�X>�S�h       �	O�<�Yc�A�*

loss o�<��K�       �	�*=�Yc�A�*

loss�R�=l�k�       �	��=�Yc�A�*

loss��Z=���       �	�W>�Yc�A�*

lossa��<$��       �	x�>�Yc�A�*

loss�>"1��       �	V�?�Yc�A�*

loss���<��j�       �	�@�Yc�A�*

loss?[�<����       �	��@�Yc�A�*

loss1ŷ<�m�       �	HA�Yc�A�*

loss\-�<�F�2       �	#B�Yc�A�*

loss��$=��Z�       �	��B�Yc�A�*

lossO��=9O�       �	yyC�Yc�A�*

loss�w)=�}Qg       �	m:D�Yc�A�*

loss�(=�R       �	gHE�Yc�A�*

loss�'=��\�       �	XF�Yc�A�*

loss�=�o��       �	&�F�Yc�A�*

loss��=��       �	XG�Yc�A�*

lossq�=�Ð�       �	�H�Yc�A�*

loss���<3t�       �	n�H�Yc�A�*

loss�~�<V�F@       �	�XI�Yc�A�*

lossL��<�ҍ�       �	�sJ�Yc�A�*

loss���;�ӑ�       �	C�K�Yc�A�*

loss-��<�}��       �	9CL�Yc�A�*

lossf�i<�;e�       �	�M�Yc�A�*

loss�q;���       �	�M�Yc�A�*

loss�#=��S       �	V}N�Yc�A�*

loss��A=�7��       �	�O�Yc�A�*

lossX��=$�6A       �	P�Yc�A�*

loss��=$z�        �	ףP�Yc�A�*

lossik�<����       �	�JQ�Yc�A�*

lossD��<JQ�       �	��Q�Yc�A�*

loss]3=fQ��       �	�|R�Yc�A�*

loss�P�<q���       �	�OS�Yc�A�*

loss�`�<@��       �	�S�Yc�A�*

loss��
<i^�        �	~�T�Yc�A�*

loss!V�;��<�       �	qU�Yc�A�*

loss��{=�й       �	��U�Yc�A�*

lossT �=BN�a       �	GWV�Yc�A�*

lossRv,=g/��       �	a�V�Yc�A�*

lossc*�;J��       �	�W�Yc�A�*

loss���;q�y       �	�WX�Yc�A�*

loss�m�<lj��       �	[�X�Yc�A�*

loss{�=,��@       �	߇Y�Yc�A�*

loss�_�;�-X�       �	CZ�Yc�A�*

loss���=PD��       �	�Z�Yc�A�*

loss�f�=a�R2       �	$�[�Yc�A�*

loss��<�G܈       �	z5\�Yc�A�*

loss:g�<T���       �	d�\�Yc�A�*

loss���:����       �	l_]�Yc�A�*

loss<��i/       �	�
^�Yc�A�*

loss�-=}��       �	B"z�Yc�A�*

loss*`�<�©�       �	��z�Yc�A�*

loss�S=j��       �	�V{�Yc�A�*

loss�^U=? ��       �	u>|�Yc�A�*

loss��<��       �	��|�Yc�A�*

loss��.<-��       �	��}�Yc�A�*

loss6`�<�ě�       �	�~�Yc�A�*

loss��	=z�>       �	�s�Yc�A�*

lossm�=��        �	�	��Yc�A�*

lossE�r=8-��       �	����Yc�A�*

lossh��;��
~       �	2=��Yc�A�*

loss�ݎ<�ѝ       �	A,��Yc�A�*

loss�CP<0e�O       �	���Yc�A�*

loss[=]2�       �	'���Yc�A�*

lossOk�=��~       �	e7��Yc�A�*

lossC�Y=�]�       �	1΄�Yc�A�*

loss���:���S       �	'i��Yc�A�*

loss�<W0�       �	��Yc�A�*

loss�?�<J�O�       �	����Yc�A�*

loss;>�=�_|       �	2��Yc�A�*

lossP�<�ЍU       �	�Ƈ�Yc�A�*

lossP�=����       �	6W��Yc�A�*

loss��&=�s�e       �	����Yc�A�*

loss1$&=�%K�       �	���Yc�A�*

loss�J=P�       �	�/��Yc�A�*

loss�<3s}       �	�ڋ�Yc�A�*

lossW�<��       �	 t��Yc�A�*

loss��0<f5^       �	���Yc�A�*

loss��<@�k       �	����Yc�A�*

loss�E}=t@&       �	�[��Yc�A�*

loss��=Ēz�       �	f���Yc�A�*

loss�B=�Y}�       �	/���Yc�A�*

loss�#U<��       �	��Yc�A�*

lossxE�=#B@A       �	鵐�Yc�A�*

loss�ѡ<~Z2       �	�V��Yc�A�*

loss��'=��`�       �	�Yc�A�*

loss�1<<f�       �	��Yc�A�*

loss�T=ϥ�(       �	����Yc�A�*

loss_J�=;ѷs       �	kE��Yc�A�*

loss�=�5�       �	)프Yc�A�*

loss���<pYi�       �	���Yc�A�*

loss��-=E4(�       �	���Yc�A�*

lossǺ�<{�1=       �	󪖄Yc�A�*

lossl��=�[A       �	�>��Yc�A�*

loss��=16D�       �	�ۗ�Yc�A�*

loss��}<� �       �	ap��Yc�A�*

loss,�6=��o�       �	���Yc�A�*

lossT��<��L       �	A���Yc�A�*

loss��E=��(G       �	{.��Yc�A�*

loss݀(;��At       �	M���Yc�A�*

loss��<�0�g       �	����Yc�A�*

loss턙=�UF�       �	G��Yc�A�*

loss̿<~K       �	����Yc�A�*

lossX��=��+V       �	1B��Yc�A�*

lossnq=5�yr       �	|ӝ�Yc�A�*

losse;���       �	v��Yc�A�*

loss\��;�H�       �	���Yc�A�*

loss�<?QR       �	����Yc�A�*

loss��<�[�v       �	����Yc�A�*

loss�x�=8��       �	�7��Yc�A�*

loss��=���       �	}ʡ�Yc�A�*

loss$�L=@�Ho       �	rm��Yc�A�*

loss�>�; .�.       �	���Yc�A�*

loss|K�=+��       �	Ҧ��Yc�A�*

loss!*=��*       �	/P��Yc�A�*

loss��<�n�y       �	P㤄Yc�A�*

loss7�A<�6r�       �	V���Yc�A�*

lossy�<�#�$       �	F[��Yc�A�*

loss(�="0{�       �	����Yc�A�*

loss��V=U�0       �	좨�Yc�A�*

loss�l�<ez$|       �	k��Yc�A�*

lossA��<���       �	���Yc�A�*

lossla�<k�4�       �	ګ��Yc�A�*

loss��<?`�       �	�D��Yc�A�*

lossn�=�l�       �	�醙Yc�A�*

loss��;����       �	˅��Yc�A�*

lossl�'=��        �	�!��Yc�A�*

loss�]U<���       �	¿��Yc�A�*

loss;�o<��O�       �	�j��Yc�A�*

loss{��=�`�`       �	��Yc�A�*

loss�6�<���e       �	����Yc�A�*

loss��:="i�       �	^J��Yc�A�*

loss]F�<Q��'       �	ᰄYc�A�*

loss l�;F;��       �	g���Yc�A�*

loss*i�<��J�       �	;��Yc�A�*

lossb��<��b�       �	TȲ�Yc�A�*

lossҸC<w⑴       �	�r��Yc�A�*

loss�G=Az�       �	M��Yc�A�*

loss�I�<���       �	�Yc�A�*

loss6��=fO9i       �	_���Yc�A�*

lossl�"=��}�       �	K:��Yc�A�*

loss�M�<�fv       �	nݶ�Yc�A�*

lossj�2=]z       �	����Yc�A�*

lossJ3�<X�'       �	���Yc�A�*

loss*y'=�Ӭ       �	�Ǹ�Yc�A�*

lossA�;Gq�:       �	���Yc�A�*

loss3�#=��ڦ       �	fM��Yc�A�*

loss}��<� �       �	(�Yc�A�*

lossl��=F��       �	�Yc�A�*

lossz="홁       �	�'��Yc�A�*

loss8)�<��g       �	�˼�Yc�A�*

loss���;�v��       �	�i��Yc�A�*

loss�dz<8��       �	����Yc�A�*

lossi��;��/g       �	`���Yc�A�*

loss#�W=vJ�       �	�2��Yc�A�*

loss�4=�G�Y       �	�տ�Yc�A�*

lossr�;<�� �       �	�|��Yc�A�*

loss[T<?���       �	���Yc�A�*

loss�=��[�       �	0���Yc�A�*

lossͶ
=/�]J       �	�zYc�A�*

loss��<���T       �	�=ÄYc�A�*

loss��<3E�       �	��ÄYc�A�*

loss�]3=3G�8       �	��ĄYc�A�*

loss�#�;�X4�       �	ńYc�A�*

loss��=.Ŧ�       �	*�ńYc�A�*

lossȊ�<��1       �	kƄYc�A�*

loss��=M�       �	VǄYc�A�*

loss3��=����       �	L�ǄYc�A�*

loss><�Nx�       �	beȄYc�A�*

lossve<.��w       �	>ɄYc�A�*

lossjcb=p�F       �	h�ɄYc�A�*

loss��i<��[       �	�SʄYc�A�*

loss�Dj=��Z*       �	�ʄYc�A�*

lossuD=�h;       �	��˄Yc�A�*

loss���;���6       �	c_̄Yc�A�*

loss#�<��5�       �	̈́Yc�A�*

loss�=6�j�       �	�̈́Yc�A�*

loss��2=Հ��       �	C΄Yc�A�*

loss��6=ݮ��       �	v�΄Yc�A�*

loss]��<�ͅ       �	e�τYc�A�*

lossĹM=��S�       �	M�ЄYc�A�*

loss�4�<Ƃ��       �	A,фYc�A�*

loss\��<F�[       �	v�фYc�A�*

loss-a�<��k�       �	�Z҄Yc�A�*

loss8a�=��d�       �	<�҄Yc�A�*

lossti�;ػ�        �	e�ӄYc�A�*

loss�͈<h��       �	F'ԄYc�A�*

loss�P2<խ��       �	'�ԄYc�A�*

loss$�<��F       �	�DքYc�A�*

lossm�<�v��       �	��քYc�A�*

loss��v;�7�       �	.tׄYc�A�*

loss;�d<��Z]       �	8؄Yc�A�*

losss�=D}�       �	C�؄Yc�A�*

loss��<�]Ь       �	�EلYc�A�*

loss�Pu=��C       �	��لYc�A�*

loss��r<��1       �	YnڄYc�A�*

loss�b�=�wVY       �	yۄYc�A�*

loss�W�:+B)       �	}�ۄYc�A�*

loss���<+�s       �	77܄Yc�A�*

loss��<'$       �	u�܄Yc�A�*

loss1<L��       �	b݄Yc�A�*

lossp�!=�Y԰       �	ބYc�A�*

loss Ƶ=O]U�       �	��ބYc�A�*

loss-��<�t�       �	�B߄Yc�A�*

loss}d=u x�       �	��Yc�A�*

loss���;F7��       �	����Yc�A�*

loss��<�V��       �	�K�Yc�A�*

lossh�v=z��       �	��Yc�A�*

loss^�=K���       �	�z�Yc�A�*

loss���;�1�B       �	��Yc�A�*

loss#�;]̂f       �	ƾ�Yc�A�*

loss�"�;��gn       �	�\�Yc�A�*

losse�M<PVN       �	���Yc�A�*

loss7�;��.�       �	į�Yc�A�*

loss5	�<~�4       �	S�Yc�A�*

loss�7�:�+�       �	M��Yc�A�*

lossiXl<o�1@       �	��Yc�A�*

loss�n<<Ɉ��       �	33�Yc�A�*

loss��<�B_�       �	A��Yc�A�*

loss}�Q<0@�6       �	���Yc�A�*

lossLA�<t�B�       �	LQ�Yc�A�*

loss,_�;$,w       �	R��Yc�A�*

loss��;_�I       �	���Yc�A�*

loss2�<r��5       �	�-�Yc�A�*

lossѵ�:���       �	��Yc�A�*

lossj�=R)p0       �	���Yc�A�*

lossQ,=�a!        �	h �Yc�A�*

loss��9=~U       �	���Yc�A�*

loss���;m|[�       �	�b�Yc�A�*

loss
b�<��Rd       �	���Yc�A�*

loss��;K貶       �	ܷ��Yc�A�*

loss��<Fc'�       �	�l�Yc�A�*

lossR��<�Y5       �	I�Yc�A�*

loss)��;yz��       �	R��Yc�A�*

loss�'�;䐗_       �	�R�Yc�A�*

loss_�1<p&       �	k��Yc�A�*

loss��$=cXB       �	��Yc�A�*

loss�E�;s�;f       �	�#��Yc�A�*

loss!7=��L       �	����Yc�A�*

loss��B=���W       �	���Yc�A�*

lossi6=�2�       �	�-��Yc�A�*

loss�5�<m��[       �	]���Yc�A�*

loss��<:��N       �	Y��Yc�A�*

lossΰ<G�(�       �	)���Yc�A�*

lossl�:�`��       �	����Yc�A�*

lossň#<�]�"       �	.p��Yc�A�*

loss�"�=��RM       �	���Yc�A�*

loss�F�=.���       �	���Yc�A�*

lossL�Y=����       �	�:��Yc�A�*

loss��;WۓT       �	N(��Yc�A�*

loss4�;���       �	6���Yc�A�*

loss�<�
m       �		l��Yc�A�*

loss���<Ae�1       �	-	��Yc�A�*

loss�9=����       �	H���Yc�A�*

loss*(<c��1       �	�G �Yc�A�*

lossS;=�2�       �	q� �Yc�A�*

lossM��=#C/�       �	���Yc�A�*

loss�m�=�|       �	��Yc�A�*

loss�.�<���       �	���Yc�A�*

loss$��=�d~       �	OY�Yc�A�*

lossD��<�!#�       �	A�Yc�A�*

loss7g)=hp��       �	X��Yc�A�*

loss%�<^ "�       �	=�Yc�A�*

loss���;5G~	       �	S �Yc�A�*

losst�<ޅ�W       �	=��Yc�A�*

lossƖ�=����       �	q��Yc�A�*

loss��>ϻ       �	e4�Yc�A�*

loss�;�=���       �	���Yc�A�*

lossV)= ͯ�       �	�	�Yc�A�*

loss��=�-*y       �	l|
�Yc�A�*

loss�7=�qd       �	y��Yc�A�*

lossa�=H��       �	j3�Yc�A�*

loss9f�<�-|�       �	�=�Yc�A�*

loss��v=��x�       �	
��Yc�A�*

loss�b;
WXh       �	���Yc�A�*

loss���=�1�3       �	�c�Yc�A�*

loss��=�L\�       �	���Yc�A�*

loss��}=XMf       �	?��Yc�A�*

loss(�<� ��       �	�C�Yc�A�*

loss���=�A��       �	Nb�Yc�A�*

loss���;L_s       �	��Yc�A�*

loss�P�;<e       �	y�Yc�A�*

lossԘR=V�M�       �	I��Yc�A�*

lossQw�=�x�>       �	�C�Yc�A�*

loss]s�;�+'       �	[��Yc�A�*

losssr3<Y�r       �	���Yc�A�*

loss��=z�K       �	@Q�Yc�A�*

loss�)/=%� �       �	�Yc�A�*

loss�.e=��b�       �	L��Yc�A�*

lossl��<�ƜW       �	_�Yc�A�*

loss�;�%3       �	���Yc�A�*

loss�#�;t�gQ       �	���Yc�A�*

loss�D=H�jG       �	5(�Yc�A�*

loss��<�ܒ�       �	���Yc�A�*

loss�K�<H/�       �	�m�Yc�A�*

lossv�=��F0       �	��Yc�A�*

loss,�=%���       �	`��Yc�A�*

lossՕ<C��       �	0H�Yc�A�*

loss��A<(�fn       �	P��Yc�A�*

loss�B_=2Ӯ�       �	��Yc�A�*

lossM9S=��       �	3 �Yc�A�*

loss�u\=D���       �	�� �Yc�A�*

lossj)�<�;��       �	y\!�Yc�A�*

loss��3<���       �	/�!�Yc�A�*

loss��<u7w�       �	�"�Yc�A�*

loss!>�;�P*       �	�%#�Yc�A�*

loss��$<�}ܷ       �	�#�Yc�A�*

loss���<�`E0       �	jg$�Yc�A�*

loss�Ӣ<����       �	ߧ%�Yc�A�*

loss���<���       �	d<&�Yc�A�*

loss��<_��X       �	,�&�Yc�A�*

loss�Ó<���D       �	u'�Yc�A�*

lossi% =��Z�       �	�(�Yc�A�*

loss�<�%x}       �	��(�Yc�A�*

loss���<�h       �	2�)�Yc�A�*

loss&,�;\QF�       �	B*�Yc�A�*

loss�/A=/P�       �	�*�Yc�A�*

lossF�E=~���       �	nk+�Yc�A�*

loss1f�=ru�C       �	�,�Yc�A�*

loss��<8}3       �	�,�Yc�A�*

loss
��<�~��       �	�2-�Yc�A�*

loss۶<�>C        �	n�-�Yc�A�*

loss�
�<C���       �	!�.�Yc�A�*

loss\=G��       �	�0/�Yc�A�*

lossM/e=���       �	?0�Yc�A�*

lossQ=�҄�       �	��0�Yc�A�*

loss�<d@D       �	�C1�Yc�A�*

loss�+;Dm��       �	X�2�Yc�A�*

loss�E0=��&       �	�]3�Yc�A�*

loss6�P=�f8g       �	� 4�Yc�A�*

lossqF�:<��       �	�4�Yc�A�*

loss{��<WY��       �	�6�Yc�A�*

lossť�<�վ2       �	�6�Yc�A�*

loss���;,Յ�       �	�^7�Yc�A�*

loss曱<��	       �	��7�Yc�A�*

loss���=
��!       �	��8�Yc�A�*

loss�f�<�_[=       �	�69�Yc�A�*

loss/��<Bڧ�       �	�9�Yc�A�*

loss#�<ݔ��       �	�l:�Yc�A�*

lossDG�<)%�        �	��:�Yc�A�*

losst�;ҩ�       �	��;�Yc�A�*

loss���;��       �	�><�Yc�A�*

lossz��;�1�       �	��<�Yc�A�*

loss�<@�@K       �	Q�=�Yc�A�*

lossc#<���       �	$&>�Yc�A�*

loss��=e���       �	;�>�Yc�A�*

lossA��<���e       �	�y?�Yc�A�*

loss� _=S�L�       �	�@�Yc�A�*

lossA�c=V�Q�       �	��@�Yc�A�*

loss
�6<�0�8       �	��A�Yc�A�*

lossw�1=bF��       �	�%B�Yc�A�*

loss}=Yc�       �	��B�Yc�A�*

loss}�=��_�       �	��C�Yc�A�*

lossS�=>��       �	��D�Yc�A�*

loss<��=5�7        �	�,E�Yc�A�*

loss��=�U��       �	:�E�Yc�A�*

loss̈́�;a~��       �	iF�Yc�A�*

loss�fO=QVz�       �	h�G�Yc�A�*

loss6�\<��:       �	�H�Yc�A�*

loss5;=����       �	uYI�Yc�A�*

lossߗ<NQ(�       �	��I�Yc�A�*

loss��<ܓWb       �	��J�Yc�A�*

loss@<<=ʩ&�       �	X;K�Yc�A�*

loss|t<���       �	O�K�Yc�A�*

lossCk�;4ט!       �	U�L�Yc�A�*

loss��[<�[S>       �	�%M�Yc�A�*

loss_؂<u�6�       �	b�M�Yc�A�*

lossR�(=W�Q       �	�cN�Yc�A�*

loss�Ri=����       �	�O�Yc�A�*

loss���=X�հ       �	ٳO�Yc�A�*

loss�=�Z��       �	#MP�Yc�A�*

loss__�=��sy       �	��P�Yc�A�*

loss<�F="�l       �	1zQ�Yc�A�*

loss���<!��       �	�R�Yc�A�*

lossT	=�S�       �	J�R�Yc�A�*

loss�rV=�O       �	�uS�Yc�A�*

loss�_<ˈT�       �	�T�Yc�A�*

loss_��<E��       �	+�T�Yc�A�*

loss��=43�       �	y?U�Yc�A�*

loss�x8=Sπ�       �	�U�Yc�A�*

loss?��;Q6        �	DjV�Yc�A�*

loss;�;���K       �	iW�Yc�A�*

lossH�7<>N��       �	�W�Yc�A�*

loss��b<��X       �	�1X�Yc�A�*

loss!Zx=4Dw�       �	fY�Yc�A�*

loss߭�<`'m�       �	�Z�Yc�A�*

lossaX=`��       �	�[�Yc�A�*

lossfJ�=�݃U       �	Lp\�Yc�A�*

loss�|K=|YP�       �	�]�Yc�A�*

lossX��;b�:5       �	�]�Yc�A�*

lossE:�;�H��       �	P^�Yc�A�*

lossi��<�UI�       �	�^�Yc�A�*

loss���=1�L	       �	��_�Yc�A�*

loss���<9�+�       �	nN`�Yc�A�*

loss�'>��?       �	��`�Yc�A�*

loss���<I�6       �	�a�Yc�A�*

loss��4=��q$       �	��b�Yc�A�*

loss���=��L�       �	�Cc�Yc�A�*

loss�6=���e       �	��c�Yc�A�*

loss�]�<�{�5       �	ߌd�Yc�A�*

loss7mG;�,�?       �	%e�Yc�A�*

lossR�;d!       �	��e�Yc�A�*

loss�V�=�,A"       �	m�f�Yc�A�*

lossQ,+<o�Y�       �	Ԟg�Yc�A�*

loss|��=*�,�       �	FAh�Yc�A�*

lossÝ�<��i       �	~i�Yc�A�*

loss�<�o�       �	Rj�Yc�A�*

lossD�<}��       �	�_k�Yc�A�*

loss�]=�[��       �	�	l�Yc�A�*

loss�UC=Ӏ~�       �	e�l�Yc�A�*

loss�i�<$`@�       �	��m�Yc�A�*

loss�r�<�u>8       �	@3n�Yc�A�*

loss�D=��ֱ       �	B�n�Yc�A�*

loss��=tN��       �	�o�Yc�A�*

loss��k=uY�       �	�p�Yc�A�*

loss.ä=mM�       �	͕r�Yc�A�*

loss
?�=o�2P       �	�is�Yc�A�*

loss�Q�<x�W�       �		t�Yc�A�*

loss�b<�9+U       �	ٴt�Yc�A�*

lossfG�=�­�       �	{Ou�Yc�A�*

lossTf=��W       �	#0v�Yc�A�*

loss�N�<O�A       �	&�v�Yc�A�*

loss��<��       �	jw�Yc�A�*

loss�y�<� �       �	�x�Yc�A�*

loss�F=���       �	<�x�Yc�A�*

loss6�7<��$        �	�Ky�Yc�A�*

lossn(=A�y�       �	��y�Yc�A�*

loss�i>=N���       �	?�z�Yc�A�*

loss��<M�N       �	Q1{�Yc�A�*

loss���<��֘       �	��{�Yc�A�*

loss���=䦀
       �	o�|�Yc�A�*

loss]�5=���-       �	-}�Yc�A�*

loss|0�=6�d#       �	��}�Yc�A�*

lossr)E=	       �	k~�Yc�A�*

loss�XP;�$��       �	B	�Yc�A�*

lossү=�}m       �	���Yc�A�*

loss�ZI=A�L       �	�M��Yc�A�*

lossUH�=�4�       �	�䀅Yc�A�*

loss��L<�R�D       �	�}��Yc�A�*

lossn��;���T       �	w��Yc�A�*

loss��<���3       �	(���Yc�A�*

losst��<��       �	����Yc�A�*

loss:��;7�U�       �	�g��Yc�A�*

loss��t=<�Pn       �	%t��Yc�A�*

loss�{�<3���       �	0��Yc�A�*

lossOt�<N�ٔ       �	5�Yc�A�*

loss�ʣ<�2Z       �	m���Yc�A�*

lossE �<���       �	�B��Yc�A�*

loss�Y�;䮭       �	�݈�Yc�A�*

loss�;�<�?��       �	͒��Yc�A�*

loss��x<dU��       �	�*��Yc�A�*

loss�=�=�ٴ|       �	Ê�Yc�A�*

loss�4=�I;�       �	�[��Yc�A�*

loss�R�<T�mb       �	m��Yc�A�*

loss��1=��8/       �	t���Yc�A�*

loss��<=�(       �	SY��Yc�A�*

loss�=q�hz       �	r���Yc�A�*

loss�=�@OC       �	>���Yc�A�*

loss���<)��c       �	>@��Yc�A�*

loss8D=}�@r       �	�폅Yc�A�*

losss��<��8�       �	���Yc�A�*

lossh�G=Gq�       �	ӑ�Yc�A�*

loss(x=|�t�       �	�s��Yc�A�*

lossF]O=\'�6       �		��Yc�A�*

loss��<�C��       �	沓�Yc�A�*

loss	��=UO       �	�l��Yc�A�*

loss��_;Y�B�       �	)��Yc�A�*

loss߬<aG�       �	����Yc�A�*

loss��%=�3�'       �	O<��Yc�A�*

lossr�<,Ȩ       �	�Ֆ�Yc�A�*

loss�$�<Ob       �	\t��Yc�A�*

lossߢA=�g��       �	
��Yc�A�*

loss}�;�-y       �	����Yc�A�*

lossx��=�2k�       �	�w��Yc�A�*

loss
�<��o�       �	���Yc�A�*

loss���=k_%       �	H���Yc�A�*

lossP
=o��       �	h?��Yc�A�*

lossF�=~�,�       �	L���Yc�A�*

loss���<P-�y       �	%x��Yc�A�*

loss.g$;���       �	���Yc�A�*

loss�
9=P�\       �	����Yc�A�*

loss���<���q       �	�j��Yc�A�*

loss%w�;��M       �	����Yc�A�*

loss�>!<3S       �	$&��Yc�A�*

loss�f=�ZЖ       �	mǠ�Yc�A�*

loss���=E�V�       �	a��Yc�A�*

lossq�=ݺ��       �	-��Yc�A�*

loss��;6��y       �	����Yc�A�*

loss�8;g�C       �	!=��Yc�A�*

loss�3<�|�       �	Z֣�Yc�A�*

loss�f=�xC       �	Bx��Yc�A�*

loss2�W=�ay�       �	z��Yc�A�*

loss\��<xX��       �	����Yc�A�*

losswbW=ƫ!1       �	�S��Yc�A�*

loss1k,<a��w       �	�6��Yc�A�*

loss��=|��       �	>��Yc�A�*

loss{��=Vq�       �	���Yc�A�*

loss&�M<Q�n
       �	NG��Yc�A�*

loss�{:=a���       �	�੅Yc�A�*

loss�"�<����       �	�z��Yc�A�*

loss9{�<	�|�       �	���Yc�A�*

loss��;����       �	ͯ��Yc�A�*

lossEK�;���?       �	����Yc�A�*

loss��V=mH�       �	���Yc�A�*

loss���=JPjn       �	9���Yc�A�*

lossH�=�X4�       �	�ܮ�Yc�A�*

lossl۩<��ך       �	�q��Yc�A�*

loss���<H�
f       �	^��Yc�A�*

loss�=�6�e       �	����Yc�A�*

loss�IC=&)Y       �	�T��Yc�A�*

lossao�<��;z       �	.���Yc�A�*

loss�B�;'ad�       �	�k��Yc�A�*

loss�;=���       �	��Yc�A�*

loss_� >#	B       �	����Yc�A�*

loss�>�<���       �	�[��Yc�A�*

loss-<���       �	H���Yc�A�*

loss �<q�]       �	����Yc�A�*

loss,J�<�J�       �	�,��Yc�A�*

loss�e<�Q�       �	iɷ�Yc�A�*

loss�<�       �	�b��Yc�A�*

loss���=~��^       �	U���Yc�A�*

lossZ�A=ՙ�       �	����Yc�A�*

loss�{=�U�&       �	/��Yc�A�*

lossK&�=k.k�       �	ͺ�Yc�A�*

loss.f�<V5�       �	�v��Yc�A�*

loss�?s;��~Q       �	�'��Yc�A�*

loss�;o��0       �	\Ǽ�Yc�A�*

loss�4=�       �	�i��Yc�A�*

loss(&�<��EK       �	���Yc�A�*

lossOV�=m�$
       �	ʥ��Yc�A�*

lossZ<#>�%NY       �	�:��Yc�A�*

lossv=i�ҥ       �	yW��Yc�A�*

loss<�v<���       �	����Yc�A�*

loss}��<=��       �	���Yc�A�*

loss`�,=�E	       �	�BYc�A�*

loss:!�<���       �	�-ÅYc�A�*

loss�9<�:�       �	~ąYc�A�*

loss���<ZVy�       �	d�ŅYc�A�*

lossl!"=Ÿ+�       �	�ƅYc�A�*

lossf)Q=�h��       �	�mǅYc�A�*

loss{�/=��
       �	�ȅYc�A�*

loss��<�tA}       �	�GɅYc�A�*

loss���;��P       �	�ɅYc�A�*

loss?<0a��       �	4�ʅYc�A�*

loss�x<����       �	�˅Yc�A�*

loss6��;�=�       �	�̅Yc�A� *

loss��/=���       �	�ͅYc�A� *

lossf�P=�^jJ       �	��΅Yc�A� *

loss}m=��+r       �	$�υYc�A� *

loss�>=�/+�       �	=aЅYc�A� *

loss� r=Rb�       �	�=хYc�A� *

loss	"<:4Od       �	��хYc�A� *

loss���<[��       �	ж҅Yc�A� *

loss�Ã<���       �	#JӅYc�A� *

loss��<29r�       �	�"ԅYc�A� *

lossfj�;eSj       �	�ԅYc�A� *

loss�3�<r�q�       �	ßՅYc�A� *

loss�f&=j۷       �	�kօYc�A� *

loss6�;V�        �	�^ׅYc�A� *

loss�d=j�H}       �	@�ׅYc�A� *

lossJ�<c̐       �	�+مYc�A� *

loss62�<S,�8       �	]lڅYc�A� *

loss�'�<{�?       �	<0ۅYc�A� *

lossR��<⣤�       �	��ۅYc�A� *

loss�I�<b�3�       �	�Y܅Yc�A� *

loss$�e=֕ 8       �	��܅Yc�A� *

loss?�< @s       �	Œ݅Yc�A� *

loss��;Z�H       �	�*ޅYc�A� *

loss��<�k�'       �	��ޅYc�A� *

lossm��<
.       �	�U߅Yc�A� *

lossV�f<����       �	��߅Yc�A� *

loss��=�ڵ       �	����Yc�A� *

loss�B<�կ$       �	��Yc�A� *

lossU��<#>F�       �	į�Yc�A� *

loss�h�=f�       �	�S�Yc�A� *

loss�լ<�w       �	j��Yc�A� *

loss��;�(�       �	���Yc�A� *

loss�δ<q��       �	-�Yc�A� *

lossԂR=h8��       �	@��Yc�A� *

loss��&<>�ֱ       �	d�Yc�A� *

loss�)<Ц\       �	��Yc�A� *

loss��<�jn�       �	L�Yc�A� *

loss��;��&       �	���Yc�A� *

loss�͖:,�n       �	�{�Yc�A� *

loss�^(=a�e�       �	��Yc�A� *

loss�P�;�]s       �	)�Yc�A� *

loss��.;�Y       �	���Yc�A� *

loss$IX<�?�       �	36�Yc�A� *

lossq�);�:�M       �	h�Yc�A� *

loss_?�<�Q׀       �	/��Yc�A� *

loss,p�:�C��       �	���Yc�A� *

loss�@:HG��       �	k'�Yc�A� *

lossJDn;��       �	���Yc�A� *

loss�C<2��n       �	�b�Yc�A� *

loss�=�O)       �	Y��Yc�A� *

loss8�:<�i!       �	����Yc�A� *

loss�x�:ٶ��       �	�6�Yc�A� *

loss�05=�^       �	���Yc�A� *

loss�"(>��h�       �	��Yc�A� *

loss�R[<�K       �	t&�Yc�A� *

loss3z>>'�L       �	Z��Yc�A� *

lossJ�!=?%hx       �	vP�Yc�A� *

loss�W=��MJ       �	&��Yc�A� *

lossc�:<��Σ       �	0���Yc�A� *

lossM�=�"�       �	�?��Yc�A� *

lossV�#=t�/4       �	@���Yc�A� *

losspr�=8ݩ       �	R~��Yc�A� *

loss0�=/��       �	 (��Yc�A� *

lossѻ^<->��       �	����Yc�A� *

loss��<F��       �	�p��Yc�A� *

loss��=,��       �	����Yc�A� *

loss�֟<�襔       �	-z��Yc�A� *

loss,_K=�cR?       �	�*��Yc�A� *

lossTS�=B֝       �	]���Yc�A� *

loss�V�=���       �	Y���Yc�A� *

loss�-T=Ō&?       �	&:��Yc�A� *

lossug�<���=       �	G��Yc�A� *

lossx�c=�pn       �	���Yc�A� *

lossA��<yVƧ       �	�i �Yc�A� *

loss�G�;���       �	2�Yc�A� *

loss��<��4       �	���Yc�A� *

loss�,=�y��       �	r�Yc�A� *

loss3;��*�       �	��Yc�A� *

losstrt;�6�Z       �	���Yc�A� *

losso��<Ѱ�       �	�H�Yc�A� *

loss]�<�pG�       �	��Yc�A� *

lossR�<g��       �	��Yc�A� *

loss��=X�",       �	l��Yc�A� *

loss�+=ݓ�       �	�`�Yc�A� *

loss�#Y;�7�m       �	��Yc�A� *

loss�Ӑ<�7�       �	���Yc�A� *

loss�f<@D�       �	�	�Yc�A� *

loss܇�:�Y��       �	�)
�Yc�A� *

loss��<o��       �	��
�Yc�A� *

loss��='�l2       �	�b�Yc�A� *

lossx;�;�~/�       �	K �Yc�A� *

lossZ�F=�8	n       �	a��Yc�A� *

loss]qk=dp5       �	�E�Yc�A� *

loss�r�<ɏ�       �	���Yc�A� *

lossQ�*<���       �	�x�Yc�A� *

loss�/�<�&"]       �	�I�Yc�A� *

loss2; <tp�!       �	�(�Yc�A� *

lossJ�C=3���       �	���Yc�A� *

loss��<�.       �	,}�Yc�A� *

lossu�=3���       �	T�Yc�A� *

loss�Y�=J��       �	���Yc�A� *

loss͉v;��E       �	`W�Yc�A� *

loss;��<V�F       �	���Yc�A� *

loss���<K��       �	]��Yc�A� *

loss���;��G       �	[%�Yc�A� *

loss�|<h��       �	Y�1�Yc�A� *

loss�={=C�*       �	v2�Yc�A� *

loss-��=�vB�       �	3�Yc�A� *

loss�Ѯ<�/��       �	�3�Yc�A� *

loss��0=ݩ�       �	[4�Yc�A� *

loss��=�&F       �	g�4�Yc�A� *

loss�a�<!��       �	�5�Yc�A� *

loss�8<x餀       �	)6�Yc�A� *

loss/ub=S�E       �	ƿ6�Yc�A� *

loss�VS=Vq��       �	�[7�Yc�A� *

loss-��<�D�]       �	68�Yc�A� *

loss�w<��F�       �	]�8�Yc�A� *

lossF,�=�4�a       �	�B9�Yc�A� *

lossl�h=��`       �	ly:�Yc�A� *

loss}#�<����       �	�;�Yc�A� *

lossM�;=�$��       �	t�;�Yc�A� *

loss!�<Z@Fo       �	�J<�Yc�A� *

loss�<�I�A       �	C�<�Yc�A� *

loss�~.<ybǒ       �	ҫ=�Yc�A� *

loss��= 
       �	�Q>�Yc�A� *

loss�g=�	��       �	��>�Yc�A� *

loss���<
��[       �	˜?�Yc�A� *

loss!�i<����       �	�7@�Yc�A� *

loss�*=�T�;       �	�@�Yc�A�!*

lossJ �<�CdJ       �	�fA�Yc�A�!*

lossߗ�;ʜBf       �	��A�Yc�A�!*

loss�><���       �	_�B�Yc�A�!*

loss��;�d��       �	�8C�Yc�A�!*

loss�	=��/       �	�C�Yc�A�!*

lossWk�=x5��       �	��D�Yc�A�!*

loss6;1�l�       �	�E�Yc�A�!*

lossl<�;��       �	V,F�Yc�A�!*

loss�C�<Q��x       �	q�F�Yc�A�!*

lossO��=�,       �	�jG�Yc�A�!*

loss���<��ɱ       �	�H�Yc�A�!*

loss/YO=�)x�       �	e�H�Yc�A�!*

lossL.+<=��T       �	�AI�Yc�A�!*

lossԑ<�g)'       �	s�I�Yc�A�!*

loss)5�=F�       �	\rJ�Yc�A�!*

loss���=SBÄ       �	�K�Yc�A�!*

loss5�=����       �	��K�Yc�A�!*

loss�H=b}�       �	O;L�Yc�A�!*

loss-��;S�u�       �	5�L�Yc�A�!*

loss
��=c���       �	wjM�Yc�A�!*

loss��=ئ�       �	�N�Yc�A�!*

lossz-=�?       �	�N�Yc�A�!*

loss�<iUm        �	�4O�Yc�A�!*

loss\U0=km�G       �	��O�Yc�A�!*

loss7r<Cװ       �	KtP�Yc�A�!*

loss��`<>��       �	$
Q�Yc�A�!*

loss���;x���       �	�Q�Yc�A�!*

loss�|�<!e�       �	2:R�Yc�A�!*

loss��G<Qi@&       �	,HS�Yc�A�!*

loss%��=Ӧe       �	"�S�Yc�A�!*

loss�p�;�&       �	"�T�Yc�A�!*

loss��h;t2,�       �	�ZU�Yc�A�!*

loss�R<�"L�       �	��U�Yc�A�!*

lossK:Hi�c       �	��V�Yc�A�!*

loss�t�<�Q       �	-#W�Yc�A�!*

loss8g�<lv�6       �	k�W�Yc�A�!*

lossq�=��*	       �	�RX�Yc�A�!*

loss�=C8x�       �	��X�Yc�A�!*

loss�W=�`�l       �	��Y�Yc�A�!*

lossh�=z'��       �	�Z�Yc�A�!*

loss��-=�&:       �	"�Z�Yc�A�!*

loss6�><�*�=       �	�\[�Yc�A�!*

loss�#�=��Z�       �	� \�Yc�A�!*

loss@X]<.L -       �	�\�Yc�A�!*

lossM�u=�&`       �	�1]�Yc�A�!*

lossZO=���       �	h�]�Yc�A�!*

lossSk=�E��       �	�l^�Yc�A�!*

loss${�<\9Tl       �	�_�Yc�A�!*

lossHS�;�*m�       �	��_�Yc�A�!*

loss4m'=0�݉       �	�d`�Yc�A�!*

lossF%=����       �	q�`�Yc�A�!*

loss��\<�+~�       �	�a�Yc�A�!*

loss�4�=)IE       �	r4b�Yc�A�!*

loss�Z=@(�       �	��b�Yc�A�!*

lossJ�;�4�       �	]nc�Yc�A�!*

loss�nT<ܓ�G       �	Fd�Yc�A�!*

loss͛�;K^��       �	��d�Yc�A�!*

loss{�<TO�A       �	�;e�Yc�A�!*

lossq94<�re�       �	M�e�Yc�A�!*

lossZh�<��       �	��f�Yc�A�!*

lossz/&<tGg       �	0.g�Yc�A�!*

loss�!�=��       �	��g�Yc�A�!*

lossƈ.;�gN       �	�uh�Yc�A�!*

lossmv�<���       �	�i�Yc�A�!*

lossC�<�vs       �	T�i�Yc�A�!*

loss���<�v^       �	Svk�Yc�A�!*

loss�0=�h�O       �	�l�Yc�A�!*

loss�E�=l1F�       �	�7m�Yc�A�!*

lossI�V<V��e       �	h�m�Yc�A�!*

loss-�D=�{�       �	ςn�Yc�A�!*

loss �a=��v        �	�%o�Yc�A�!*

loss��p=.�(       �	2�o�Yc�A�!*

loss�l�<=�Y       �	0�p�Yc�A�!*

loss��P<cO�       �	:"q�Yc�A�!*

loss�4�=˂
8       �	�q�Yc�A�!*

loss湁<U��<       �	Aar�Yc�A�!*

loss�g�;|P=j       �	P�r�Yc�A�!*

loss^{�;&悒       �	S�s�Yc�A�!*

loss�W�<cMS       �	�1t�Yc�A�!*

loss�<��       �	�t�Yc�A�!*

loss��<���       �	�uu�Yc�A�!*

loss(�=u>�[       �	�v�Yc�A�!*

loss~�<C�:       �	�v�Yc�A�!*

loss8�<c�%       �	qvw�Yc�A�!*

loss�=�W"       �	 x�Yc�A�!*

loss��=Nw��       �	J�x�Yc�A�!*

loss�G�<���u       �	oJy�Yc�A�!*

loss�<=����       �	�%z�Yc�A�!*

loss��=�ta       �	d�z�Yc�A�!*

loss���;�y�Q       �	Ί{�Yc�A�!*

loss94�<s���       �	34|�Yc�A�!*

loss�fa;��JE       �	��|�Yc�A�!*

loss��\<���J       �	9�}�Yc�A�!*

loss��=GHF�       �	��~�Yc�A�!*

loss��<�u�       �	�)�Yc�A�!*

loss��;;�b6       �	���Yc�A�!*

loss��=�@6       �	N~��Yc�A�!*

lossu�<#��W       �	�'��Yc�A�!*

lossja�=ns8�       �	<���Yc�A�!*

loss���<
���       �	_\��Yc�A�!*

lossW�4<nC�       �	b�Yc�A�!*

loss��)=�SIB       �	!���Yc�A�!*

lossXi= �&�       �	,+��Yc�A�!*

loss��<'Au�       �	�Ä�Yc�A�!*

lossX�d=�       �	�\��Yc�A�!*

loss�]�<NT�B       �	�2��Yc�A�!*

loss��d;�l�       �	Ɇ�Yc�A�!*

loss:�<��j�       �	���Yc�A�!*

loss�+�;�-1�       �	kG��Yc�A�!*

loss���<�}��       �	z∆Yc�A�!*

loss��<;�       �	�v��Yc�A�!*

lossR�;�Jd�       �	�U��Yc�A�!*

lossa	=Ӆ �       �	����Yc�A�!*

loss���;���       �	����Yc�A�!*

lossw��;�ve       �	qW��Yc�A�!*

lossɅ=�Qn       �	$��Yc�A�!*

loss*�:|�W       �	j���Yc�A�!*

loss=6�<Vt�       �	�N��Yc�A�!*

loss�/= ��       �	6珆Yc�A�!*

loss	E�<�;o�       �	h���Yc�A�!*

loss-�=�s��       �	�8��Yc�A�!*

loss�h�<r�5w       �	�ݑ�Yc�A�!*

lossWX�<0�	       �	�~��Yc�A�!*

loss�hK<�׮       �	#��Yc�A�!*

loss��v<�Q$       �	����Yc�A�!*

loss�n=A��B       �	�a��Yc�A�!*

loss���:'���       �	����Yc�A�!*

loss/-�<N�.       �	K���Yc�A�"*

loss�ͦ<�h8�       �	S%��Yc�A�"*

loss\Z�<Ǖ�_       �	���Yc�A�"*

loss�
=�s��       �	6���Yc�A�"*

loss	�N;���       �	�I��Yc�A�"*

loss��x<סU       �	�䘆Yc�A�"*

loss�]<=堶       �	����Yc�A�"*

loss�D=�m�3       �	���Yc�A�"*

lossv	�<���       �	ȳ��Yc�A�"*

lossa�=2T�       �	�L��Yc�A�"*

loss�b�<�6��       �	�蛆Yc�A�"*

lossD=�D-�       �	���Yc�A�"*

loss���<T��x       �	�$��Yc�A�"*

loss�I2=�~!       �	����Yc�A�"*

loss��<�M�       �	�M��Yc�A�"*

loss�$�<��^�       �	�➆Yc�A�"*

loss���<�4��       �	�y��Yc�A�"*

loss�<���       �	b��Yc�A�"*

lossvs�<��6�       �	i���Yc�A�"*

loss��=�Q�       �	�>��Yc�A�"*

loss �Q<�i��       �	vޡ�Yc�A�"*

lossq=>�UM       �	����Yc�A�"*

lossc�F<<��%       �	�G��Yc�A�"*

lossu�;���q       �	FYc�A�"*

loss�i�<�[�7       �	����Yc�A�"*

loss1�]<�l��       �	� ��Yc�A�"*

lossC>U=�l       �	򶥆Yc�A�"*

loss��M<���R       �	lZ��Yc�A�"*

loss�:\qs       �	�Yc�A�"*

loss,�;:$��       �	����Yc�A�"*

loss,8�<����       �	�4��Yc�A�"*

loss(�m<1j�       �	_Ψ�Yc�A�"*

loss�=8�G=       �	�p��Yc�A�"*

loss���<�u�b       �	�S��Yc�A�"*

loss-:fǶT       �	N�Yc�A�"*

loss1=K���       �	���Yc�A�"*

loss�O�;p��1       �	���Yc�A�"*

loss���;��G        �	v¬�Yc�A�"*

loss_�;���       �	�V��Yc�A�"*

lossT��=EM0I       �	1צּYc�A�"*

loss=��<�8��       �	4���Yc�A�"*

loss�E<�k�Y       �	@0��Yc�A�"*

loss�z�<8�O       �	>˯�Yc�A�"*

loss,��;'j�       �	�g��Yc�A�"*

lossG��<���       �	* ��Yc�A�"*

lossx�;0��       �	����Yc�A�"*

loss��l<d&W+       �	�*��Yc�A�"*

loss���<W���       �	'���Yc�A�"*

loss�y�;d+��       �	hz��Yc�A�"*

lossܗ<nmT=       �	���Yc�A�"*

loss��<��gQ       �	C���Yc�A�"*

loss�J=v�`       �	�B��Yc�A�"*

loss�VQ=���7       �	�׵�Yc�A�"*

loss:DT;zbR{       �	�y��Yc�A�"*

loss��<%� 4       �	���Yc�A�"*

loss�"=p0J       �	VD��Yc�A�"*

loss���<��D       �	�ܸ�Yc�A�"*

loss��<\1       �	�x��Yc�A�"*

loss\��=���       �	���Yc�A�"*

loss{FT<#�i�       �	[���Yc�A�"*

loss�ȧ;.�7       �	�M��Yc�A�"*

loss�};c�9Y       �	�细Yc�A�"*

lossv5<��>�       �	���Yc�A�"*

loss?4$=�$��       �	�,��Yc�A�"*

loss���<��7�       �	�Ƚ�Yc�A�"*

loss*�(>(*s�       �	Ac��Yc�A�"*

loss��=��       �	q��Yc�A�"*

lossԯ�=C��(       �	�Ͽ�Yc�A�"*

lossS8=(6�z       �	h��Yc�A�"*

lossJ��<M�G       �	����Yc�A�"*

loss�< B�       �	���Yc�A�"*

loss�ګ;a��v       �	�=Yc�A�"*

lossn�l=�_       �	y�Yc�A�"*

loss�|�<���       �	
�ÆYc�A�"*

lossΡZ=�]b�       �	�ĆYc�A�"*

lossm�:=
�(�       �	�ĆYc�A�"*

loss8��<�Y1"       �	�`ņYc�A�"*

loss�d�<�        �	��ņYc�A�"*

loss�|�=;\4�       �	CǆYc�A�"*

loss�~ <��       �	T�ǆYc�A�"*

lossm� <+�f       �	k�ȆYc�A�"*

loss��o=���       �	�iɆYc�A�"*

loss���<�=K       �	PnʆYc�A�"*

loss�%=>���       �	�&ˆYc�A�"*

loss�<��
�       �	��ˆYc�A�"*

loss��Y<vZ��       �	�̆Yc�A�"*

loss���<@%�       �	a�͆Yc�A�"*

loss6�,= �;k       �	�'ΆYc�A�"*

loss-h�<Z��        �	�φYc�A�"*

losso�=�jD       �	�ІYc�A�"*

loss��<�Y�       �	��ІYc�A�"*

loss�Y�;F��       �	HцYc�A�"*

loss��<%>�       �	A�цYc�A�"*

lossI�<v���       �	L�҆Yc�A�"*

loss�T=��.�       �	;4ӆYc�A�"*

lossc=%;�
E�       �	g�ӆYc�A�"*

loss�cU<X2ٺ       �	��ԆYc�A�"*

loss���=�4�       �	�"ՆYc�A�"*

loss�Pk<��       �	5�ՆYc�A�"*

loss�}=�v�       �	IֆYc�A�"*

loss-�M<r�=       �	��ֆYc�A�"*

loss��=��7�       �	�x׆Yc�A�"*

loss�+c<%��       �	�؆Yc�A�"*

loss�=H���       �	�؆Yc�A�"*

loss��C=7\�P       �	CنYc�A�"*

loss=��<��u       �	��نYc�A�"*

loss��^<a�t�       �	k�چYc�A�"*

loss)��<�
Ǆ       �	nۆYc�A�"*

losshPt<i�C       �	%�ۆYc�A�"*

loss��<MB       �	@L܆Yc�A�"*

loss��<~d4�       �	��܆Yc�A�"*

loss�%`=N&�Y       �	�y݆Yc�A�"*

loss;
�<ЄC�       �	�ކYc�A�"*

loss� ,<y��       �	��ކYc�A�"*

loss��;�G�1       �	,J߆Yc�A�"*

loss{�#=�gڙ       �	�߆Yc�A�"*

loss��;� �M       �	�y��Yc�A�"*

loss� �<�8@]       �	�Yc�A�"*

loss˃=Ko�a       �	P��Yc�A�"*

loss;=�.};       �	�]�Yc�A�"*

lossϰ_=�h�c       �	���Yc�A�"*

loss�<b��       �	x��Yc�A�"*

loss[�=!+/5       �	v6�Yc�A�"*

lossqg�<�L4       �	���Yc�A�"*

loss�=�<if��       �	Uh�Yc�A�"*

loss=��<;gPn       �	��Yc�A�"*

losst{�<�"�       �	k��Yc�A�"*

lossZި;����       �	&5�Yc�A�"*

loss:��<h���       �	S��Yc�A�#*

loss��o;��^       �	�i�Yc�A�#*

loss7�<�L       �	:�Yc�A�#*

loss���<&i��       �	0��Yc�A�#*

loss�F=��R`       �	�@�Yc�A�#*

loss��<,t'�       �	���Yc�A�#*

loss��;�m(f       �	�v�Yc�A�#*

loss<�='KVK       �	��Yc�A�#*

loss�5=|`v       �	���Yc�A�#*

loss��=���       �	�N�Yc�A�#*

loss(2�<鴣u       �	��Yc�A�#*

loss��;�J�S       �	C��Yc�A�#*

loss��c=1��       �	I�Yc�A�#*

lossm
=���       �	���Yc�A�#*

lossdà;x9�       �	�|��Yc�A�#*

lossZ&�;�m�m       �	��Yc�A�#*

loss�`�=�%s�       �	���Yc�A�#*

loss�I�;|{8U       �	�L�Yc�A�#*

loss�"= Ko�       �	���Yc�A�#*

loss@��<MD.\       �	���Yc�A�#*

loss�<gk�l       �	��Yc�A�#*

losstB"<���       �	���Yc�A�#*

loss��6<	� B       �	/���Yc�A�#*

losshR�;NMA       �	�;��Yc�A�#*

loss&2�=�FK�       �	���Yc�A�#*

loss
�=3��       �	�k��Yc�A�#*

loss�9=���       �	l��Yc�A�#*

lossW`�:yS@       �	����Yc�A�#*

loss6m�;�}oV       �	O;��Yc�A�#*

lossS?<I�n�       �	����Yc�A�#*

lossj�=�       �	�u��Yc�A�#*

loss�U�<`�]�       �	�[��Yc�A�#*

loss  l=P�N�       �	(���Yc�A�#*

loss���={��b       �	����Yc�A�#*

loss7xs;�s�       �	>!��Yc�A�#*

loss-��;���       �	8���Yc�A�#*

loss���<
Ր�       �	>]��Yc�A�#*

loss��;�       �	@���Yc�A�#*

loss�M�<��       �	����Yc�A�#*

loss��;=���       �	J( �Yc�A�#*

loss�1<�}��       �	L� �Yc�A�#*

lossߊ<ń%�       �	�]�Yc�A�#*

lossO��<�vK�       �	���Yc�A�#*

loss��<�ӫ�       �		��Yc�A�#*

loss7�B;�a�       �	l#�Yc�A�#*

loss)}M=c�	�       �	���Yc�A�#*

loss���=��d�       �	3O�Yc�A�#*

loss��<2��b       �	T��Yc�A�#*

lossv�;�#	       �	s�Yc�A�#*

lossi/�<�z       �	o�Yc�A�#*

loss2�)=���e       �	�S�Yc�A�#*

loss�K�;�qXo       �	��Yc�A�#*

lossF�<���       �	���Yc�A�#*

loss�/C;�
[:       �	�z	�Yc�A�#*

loss��$=(r(O       �	U�
�Yc�A�#*

loss�o�=�X�       �	W�Yc�A�#*

loss�[<Oc��       �	�M�Yc�A�#*

loss�~�<�o��       �	���Yc�A�#*

loss�<:��       �	���Yc�A�#*

loss<S=l&��       �	��Yc�A�#*

loss���;���V       �	�k�Yc�A�#*

loss�y%=ɍ�       �	j�Yc�A�#*

loss��<�<u       �	���Yc�A�#*

loss΢<����       �	�d�Yc�A�#*

lossZ��<1Qï       �	'�Yc�A�#*

loss�V=���       �	�<�Yc�A�#*

lossfð=8�EI       �	��Yc�A�#*

loss�d�;��&       �	�Yc�A�#*

loss��=�       �	���Yc�A�#*

loss�A=3h�       �	Po�Yc�A�#*

loss��<�uM       �	=)�Yc�A�#*

lossM��<�.��       �	���Yc�A�#*

loss�@�<����       �	i�Yc�A�#*

loss<�<'l�       �	N�Yc�A�#*

loss q1;n�       �	���Yc�A�#*

loss���<b��       �	�e�Yc�A�#*

loss���;�ǉ       �	 �Yc�A�#*

loss؁;�"�       �	��Yc�A�#*

loss ��<!�k       �	�E�Yc�A�#*

loss60�<)���       �	���Yc�A�#*

loss�\:�\Ԫ       �	"��Yc�A�#*

loss�f�=d
+7       �	�'�Yc�A�#*

lossG�=Kp0�       �	2 �Yc�A�#*

loss�}<;�!       �	�� �Yc�A�#*

losslG(=6m       �	�i!�Yc�A�#*

loss��%=���y       �	�"�Yc�A�#*

lossj��<�9�;       �	��"�Yc�A�#*

loss���=t���       �	�q#�Yc�A�#*

loss��<N���       �	�$�Yc�A�#*

loss7��;.`N       �	z�$�Yc�A�#*

loss��<w�@E       �	P%�Yc�A�#*

lossƐ�;!�q�       �	q�%�Yc�A�#*

loss��;�$��       �	��&�Yc�A�#*

lossX�m=��ʌ       �	�\'�Yc�A�#*

lossj3<�T�_       �	�(�Yc�A�#*

loss���;��aD       �	��(�Yc�A�#*

loss�=�� �       �	D)�Yc�A�#*

loss�/�;���&       �	��)�Yc�A�#*

loss�t=���       �	 q*�Yc�A�#*

loss��e<R�W�       �	B+�Yc�A�#*

loss�ma=��W       �	<�+�Yc�A�#*

loss�6<�ѻ"       �	�7,�Yc�A�#*

loss���<�i|       �	��,�Yc�A�#*

loss�%�=(se�       �	�e-�Yc�A�#*

lossf7*<�}       �	��-�Yc�A�#*

loss�<�;8N��       �	B�.�Yc�A�#*

loss!��=�%:       �	�//�Yc�A�#*

loss��p=q��       �	��/�Yc�A�#*

loss]7>��       �	�d0�Yc�A�#*

loss\�b;b�Ь       �	 1�Yc�A�#*

lossG=E{�       �	��1�Yc�A�#*

loss �\<+d.       �	�F2�Yc�A�#*

lossM"
=���       �	��2�Yc�A�#*

lossz\[<�oޕ       �	�r3�Yc�A�#*

lossC��<�_=       �	�4�Yc�A�#*

loss�$e<����       �	��4�Yc�A�#*

loss�Ӌ=�XTR       �	�65�Yc�A�#*

loss�*<�I<�       �	��5�Yc�A�#*

loss <=��       �	�c6�Yc�A�#*

lossl<U���       �	.77�Yc�A�#*

loss�%�<Y�d�       �	��7�Yc�A�#*

loss��*<��Q       �	qr8�Yc�A�#*

loss�h�<�
��       �	w.9�Yc�A�#*

loss,�]<���       �	��9�Yc�A�#*

loss<=9��Y       �	��;�Yc�A�#*

loss��_=�n�       �	�T=�Yc�A�#*

lossM�r<���       �	��=�Yc�A�#*

lossE۳=�s=�       �	�>�Yc�A�#*

loss�3<l��U       �	&?�Yc�A�$*

loss9 =���S       �	5�?�Yc�A�$*

loss�Q�=�<       �	DN@�Yc�A�$*

lossE�=2�%       �	�@�Yc�A�$*

lossѥ�=���       �	��A�Yc�A�$*

loss2:�;���@       �	w+B�Yc�A�$*

loss8�=�(��       �	��B�Yc�A�$*

loss�q)=؝c�       �	uXC�Yc�A�$*

loss�D=�Z��       �	�C�Yc�A�$*

lossu�;���       �	��D�Yc�A�$*

loss%�:�L�i       �	>!E�Yc�A�$*

lossV@�<wE�a       �	��E�Yc�A�$*

loss�K=��       �	J�F�Yc�A�$*

loss:k�<tս�       �	��G�Yc�A�$*

loss���=#U��       �	8�H�Yc�A�$*

loss_�6<P�)       �	�MI�Yc�A�$*

loss&X�=�?��       �	<J�Yc�A�$*

loss�:=���       �	b�J�Yc�A�$*

loss��=�X�       �	��K�Yc�A�$*

loss��<G�P�       �	�_L�Yc�A�$*

loss���=�V@�       �	. M�Yc�A�$*

loss<w�<<�"       �	&N�Yc�A�$*

lossN�V<���`       �	�>O�Yc�A�$*

loss�� =}���       �	��O�Yc�A�$*

loss�l=�j0       �	 sP�Yc�A�$*

losso(�<3�       �	Q�Yc�A�$*

loss�$�<r8.�       �	D�Q�Yc�A�$*

loss�F=��w       �	�=R�Yc�A�$*

lossF��<���3       �	��R�Yc�A�$*

loss�H�;/��       �	u�S�Yc�A�$*

loss�3=�+�       �	�_T�Yc�A�$*

loss��o<g��}       �	�U�Yc�A�$*

loss�bo<زD�       �	��U�Yc�A�$*

loss�l^=��t[       �	c)V�Yc�A�$*

loss��=�M�       �	�V�Yc�A�$*

lossX�M=�0�f       �	�OW�Yc�A�$*

lossD9�<Y=�       �	��W�Yc�A�$*

loss��-;\E�       �	8�X�Yc�A�$*

loss7B�<�m�       �	HQY�Yc�A�$*

losssg�=�k��       �	��Y�Yc�A�$*

lossP<���k       �	�{Z�Yc�A�$*

loss�Pu<�'x�       �	@[�Yc�A�$*

loss3��=)��       �	�[�Yc�A�$*

loss��R=
v�       �	~7\�Yc�A�$*

loss��<)�       �	��\�Yc�A�$*

loss�
<��~@       �	|c]�Yc�A�$*

lossNt<����       �	w�]�Yc�A�$*

loss�/�=9��       �	c�^�Yc�A�$*

loss��<�U&�       �	�*_�Yc�A�$*

lossl��<��K�       �	H�_�Yc�A�$*

lossw��<ñ�0       �	�X`�Yc�A�$*

loss�2�<t��t       �	�`�Yc�A�$*

loss��<       �	"�a�Yc�A�$*

loss %6=��]~       �	�b�Yc�A�$*

lossry~;�y~       �	��b�Yc�A�$*

lossiy�;�k'       �	�Wc�Yc�A�$*

loss�F>�7]       �	��c�Yc�A�$*

lossn��<^�ĉ       �	\�d�Yc�A�$*

loss_�;)�S�       �	'e�Yc�A�$*

loss4�%<O�2^       �	+�e�Yc�A�$*

loss��I<�*b�       �	�[f�Yc�A�$*

loss��];4�9r       �	b�f�Yc�A�$*

loss�;�a��       �	�g�Yc�A�$*

loss�y�=<�n       �	�(h�Yc�A�$*

loss��<�؛�       �	��h�Yc�A�$*

loss��4=ta(       �	�\i�Yc�A�$*

lossF !=?���       �	��i�Yc�A�$*

loss�u*<-;�       �	s�j�Yc�A�$*

loss�ݗ<ŗ>       �	&:k�Yc�A�$*

loss@,J;�-��       �	$�k�Yc�A�$*

loss��=_���       �	��l�Yc�A�$*

loss�[=�@��       �	{�m�Yc�A�$*

loss�>)|�       �	�5n�Yc�A�$*

loss�a?=��8,       �	��n�Yc�A�$*

lossH�<�h�o       �	�lo�Yc�A�$*

loss�p�;c��       �	v�o�Yc�A�$*

lossS�<�W�       �	0�p�Yc�A�$*

lossq�=8�d{       �	jq�Yc�A�$*

lossn�Y;#�       �	�r�Yc�A�$*

loss���;�b�       �	Q�r�Yc�A�$*

loss�:`= �%{       �	��s�Yc�A�$*

lossfh@;����       �	�mt�Yc�A�$*

loss�<��(�       �	u�Yc�A�$*

loss,ؖ<��6�       �	��u�Yc�A�$*

loss��t<�3�       �	i8v�Yc�A�$*

loss�r�;�=�       �	��v�Yc�A�$*

loss��;�X�2       �	U�w�Yc�A�$*

loss�s�;"³       �	Cx�Yc�A�$*

loss4��=����       �	�x�Yc�A�$*

loss�O<��5       �	z�Yc�A�$*

loss�I<�H       �	N�z�Yc�A�$*

loss�<�"��       �	�5{�Yc�A�$*

loss���=S�.       �	��{�Yc�A�$*

loss�E=���       �	p|�Yc�A�$*

loss�ʺ<�\I       �	I}�Yc�A�$*

loss
�n<؏�       �	%�}�Yc�A�$*

loss�&<M��       �	:�~�Yc�A�$*

loss���=�Q       �	�/�Yc�A�$*

loss��<.�Ӱ       �	X��Yc�A�$*

loss�l�<��ԙ       �	�^��Yc�A�$*

loss	�;��#�       �	����Yc�A�$*

lossT�0=�k�k       �	���Yc�A�$*

loss
S�;��rv       �	j.��Yc�A�$*

loss��<��h       �	Kɂ�Yc�A�$*

loss��Y=�
�       �	�`��Yc�A�$*

losscU�<\��       �	���Yc�A�$*

loss�z*=�(�       �	����Yc�A�$*

loss�<P���       �	�,��Yc�A�$*

loss#.�<E�K       �	�Ņ�Yc�A�$*

losszL=أ�       �	�_��Yc�A�$*

loss�ƿ<�܀-       �	����Yc�A�$*

loss���;���       �	����Yc�A�$*

loss(��<���       �	�f��Yc�A�$*

lossͬ�;f~p       �	7��Yc�A�$*

loss(v�;o���       �	�n��Yc�A�$*

loss��:ټ-�       �	i���Yc�A�$*

losscݓ=���       �	k��Yc�A�$*

loss�\�=r[��       �	�&��Yc�A�$*

loss�C�<Ni!�       �	덇Yc�A�$*

loss�j<���       �	dώ�Yc�A�$*

loss:C'=�"��       �	�s��Yc�A�$*

lossZx�=�	0�       �	���Yc�A�$*

loss���=+�//       �	���Yc�A�$*

loss�|�;m��w       �	oF��Yc�A�$*

lossr<Wy��       �	 䑇Yc�A�$*

loss
/F;��       �	�}��Yc�A�$*

loss��;��       �	���Yc�A�$*

loss	�;��؏       �	ū��Yc�A�$*

loss�ͨ;Q��       �	kD��Yc�A�%*

loss�>x:�]�k       �	����Yc�A�%*

lossr-D<��Fu       �	F���Yc�A�%*

loss�ϗ;��K�       �	sH��Yc�A�%*

lossr� =���       �	CᖇYc�A�%*

loss!�><���       �	k���Yc�A�%*

loss�'s;gD��       �	a��Yc�A�%*

loss��P;Hy#3       �	Թ��Yc�A�%*

lossMߣ<��u�       �	�V��Yc�A�%*

loss��=y�U�       �	���Yc�A�%*

loss�	<���       �	Ϣ��Yc�A�%*

lossh:�_       �	�U��Yc�A�%*

loss�|<U�{       �	�ꛇYc�A�%*

lossC� >`,1        �	����Yc�A�%*

loss��;����       �	H��Yc�A�%*

loss�~ >����       �	����Yc�A�%*

loss	v=��       �	�J��Yc�A�%*

loss��w<%Jt       �	L㞇Yc�A�%*

loss�#<�i��       �	Bz��Yc�A�%*

loss-}�<@kn�       �	j��Yc�A�%*

loss��<Ux�0       �	���Yc�A�%*

loss�	�<��o       �	bJ��Yc�A�%*

loss.=gj��       �	硇Yc�A�%*

loss}К<�p�       �	�{��Yc�A�%*

loss� w<hB�       �	'��Yc�A�%*

loss��=0�!�       �	����Yc�A�%*

lossLe<m�,       �	D��Yc�A�%*

lossrS�;8       �	�פ�Yc�A�%*

lossS�<1��       �	�n��Yc�A�%*

loss	V�=�k��       �	)��Yc�A�%*

losst�>=��	n       �	X���Yc�A�%*

lossRT�<{ʱ       �	tD��Yc�A�%*

lossS'�<�V�       �	l짇Yc�A�%*

lossw��<.���       �	m���Yc�A�%*

lossv�'<&6z�       �	+5��Yc�A�%*

lossu=�/��       �	ש�Yc�A�%*

loss�5=RFb�       �	�q��Yc�A�%*

loss�@�;(�g       �	S��Yc�A�%*

lossn�@;\hڠ       �	^���Yc�A�%*

loss�!<I��o       �	'1��Yc�A�%*

loss�wJ=��C       �	�Ǭ�Yc�A�%*

lossIl�<	[@�       �	)_��Yc�A�%*

lossH#�=��0       �	��Yc�A�%*

lossDt�=nm��       �	׊��Yc�A�%*

lossć�<� �
       �	���Yc�A�%*

loss`��<�z�       �	K���Yc�A�%*

loss�'<t���       �	�\��Yc�A�%*

loss�@�:<�       �	�	��Yc�A�%*

lossi��<g!*       �	���Yc�A�%*

loss�0<.��       �	g_��Yc�A�%*

loss�n�;�Mr        �	��Yc�A�%*

loss\�E=�c;       �	5���Yc�A�%*

lossa�0=N.'�       �	�[��Yc�A�%*

loss(v8<b48^       �	�%��Yc�A�%*

lossA�K<G%'       �	d˵�Yc�A�%*

loss3��=����       �	m��Yc�A�%*

loss�<tk��       �	2��Yc�A�%*

loss5M�<Pm9       �	Yc�A�%*

loss}
�;�36�       �	'���Yc�A�%*

loss+:=�X�       �	�L��Yc�A�%*

loss�9=C��a       �	���Yc�A�%*

loss;O<s�       �	񹺇Yc�A�%*

loss��<9�qr       �	�a��Yc�A�%*

loss�B};䏖       �	����Yc�A�%*

loss�v[;}G��       �	g���Yc�A�%*

loss��6<Ł�       �	��ԇYc�A�%*

lossj�=u��       �	'iՇYc�A�%*

lossd�=k��       �	�ևYc�A�%*

lossӨ=��%       �	k�ևYc�A�%*

loss܎=���       �	�4ׇYc�A�%*

lossv�<C��g       �	v�ׇYc�A�%*

loss��	=�sw       �	�|؇Yc�A�%*

loss+��<ubK}       �	OهYc�A�%*

loss�t=�4       �	�هYc�A�%*

loss ��<c[B�       �	>\ڇYc�A�%*

losss§<�b"	       �	<�ڇYc�A�%*

loss��</�K�       �	#�ۇYc�A�%*

loss�I=�G۵       �	�M܇Yc�A�%*

loss3o�=���*       �	.�܇Yc�A�%*

loss��=���       �	N݇Yc�A�%*

loss�;P]�W       �	HއYc�A�%*

loss�ؒ:8��       �	|�އYc�A�%*

loss�H�;��%       �	�!�Yc�A�%*

lossM�=f�t�       �	���Yc�A�%*

lossq��<�)�l       �	��Yc�A�%*

loss�	L<���       �	�(�Yc�A�%*

loss�0�=2��       �	s��Yc�A�%*

loss�u�:�;��       �	6W�Yc�A�%*

loss�r�<<�x       �	V��Yc�A�%*

lossÜ�<��7k       �	 ��Yc�A�%*

loss�	�<�O5�       �	�-�Yc�A�%*

lossrs)=��       �	��Yc�A�%*

loss�SI<���1       �	Y�Yc�A�%*

lossn��;�Vy�       �	���Yc�A�%*

loss6��<�� �       �	ˀ�Yc�A�%*

loss��<^���       �	��Yc�A�%*

loss�x�;�T�       �	3��Yc�A�%*

lossE��<�汪       �	�@�Yc�A�%*

loss�yP=)�<�       �	���Yc�A�%*

loss_� =���S       �	9|�Yc�A�%*

loss�0q=Dy�f       �	K�Yc�A�%*

loss�8;�i�p       �	3��Yc�A�%*

loss���<�8�
       �	Q��Yc�A�%*

loss,C1=�m�       �	�9�Yc�A�%*

lossZR�<�       �	��Yc�A�%*

loss}�
;n��       �	]l�Yc�A�%*

loss�F,=3O�        �	���Yc�A�%*

losso�)<�o0       �	e���Yc�A�%*

loss���<ƣ�       �	K?�Yc�A�%*

lossw�<α�Q       �	o��Yc�A�%*

loss.��<�S       �	s�Yc�A�%*

loss�/=<BV       �	F�Yc�A�%*

lossA(�<�{�       �	8��Yc�A�%*

loss�L�<���       �	�5�Yc�A�%*

loss��:+�H       �	���Yc�A�%*

loss3v�;�ir       �	�v��Yc�A�%*

loss�35=��t�       �	���Yc�A�%*

loss�]�;���r       �	���Yc�A�%*

loss��^=���o       �	�8��Yc�A�%*

loss�@�<�<�       �	����Yc�A�%*

lossB�<�E�<       �	f��Yc�A�%*

loss�2~<�Z�t       �	����Yc�A�%*

lossq�:��!\       �	C���Yc�A�%*

loss�&=���x       �	A,��Yc�A�%*

loss/S"=b       �	����Yc�A�%*

loss��D>���       �	�Y��Yc�A�%*

loss���:X(��       �	����Yc�A�%*

loss�Q�;�I�       �	����Yc�A�%*

lossHK=��G�       �	�&��Yc�A�&*

loss��;�RK�       �	���Yc�A�&*

lossNl;��y$       �	X��Yc�A�&*

loss[͐=�p�       �	����Yc�A�&*

loss�U�<=�e       �	ݗ��Yc�A�&*

loss,��<�O�p       �	0 �Yc�A�&*

loss��=�\�       �	�� �Yc�A�&*

loss��D<d�2       �	�b�Yc�A�&*

loss*�W<��T�       �	r��Yc�A�&*

loss��<#�Z�       �	���Yc�A�&*

loss��=M{�       �	.�Yc�A�&*

loss<5�=�*4�       �	���Yc�A�&*

loss��;!ߺ7       �	�i�Yc�A�&*

loss��:�g��       �	�Yc�A�&*

loss݄�=#b�       �	z��Yc�A�&*

loss�=;<
�?Z       �	
I�Yc�A�&*

loss��Y<a&8       �	v��Yc�A�&*

loss�$=c��$       �	1��Yc�A�&*

loss�>�<!];Z       �	$|�Yc�A�&*

loss�U�<8<aq       �	�+	�Yc�A�&*

lossfq<	��V       �	��	�Yc�A�&*

loss�Ռ<M>�       �	ni
�Yc�A�&*

loss��%=�04       �	
�Yc�A�&*

loss�;�!��       �	y��Yc�A�&*

loss��=,X�"       �	\W�Yc�A�&*

loss[b$=�I+       �	���Yc�A�&*

loss���;�xV       �	I��Yc�A�&*

loss�5=Ӹ8L       �	�9�Yc�A�&*

lossҁ=p�       �	���Yc�A�&*

lossN�<��|       �	�y�Yc�A�&*

loss;�%<� _�       �	! �Yc�A�&*

loss��P<�G�K       �	���Yc�A�&*

losscT�<��       �	�j�Yc�A�&*

loss��P=q 5       �	0�Yc�A�&*

lossa�[<5�&�       �	���Yc�A�&*

loss4�2<X��       �	hY�Yc�A�&*

lossZ�%<��*       �	� �Yc�A�&*

loss�5�<U��,       �	���Yc�A�&*

loss]�x;cJ$       �	=I�Yc�A�&*

loss�e<��O�       �	���Yc�A�&*

loss�'<��E@       �	F��Yc�A�&*

loss�yy=Ɉ>        �	�8�Yc�A�&*

loss%�R=g��(       �	$��Yc�A�&*

loss�V;��U8       �	�y�Yc�A�&*

loss�,<#�T       �	/�Yc�A�&*

loss�4=��I%       �	
��Yc�A�&*

loss�)==��       �	�T�Yc�A�&*

loss�|=01XH       �	B�Yc�A�&*

loss��H<}��       �	��Yc�A�&*

loss#T<w��       �	�Y�Yc�A�&*

loss �:�       �	'.�Yc�A�&*

loss-=�\O�       �	T��Yc�A�&*

loss m<H��       �	؞�Yc�A�&*

loss��<�4�       �	�A �Yc�A�&*

loss=HM�)       �	T� �Yc�A�&*

loss}��<��e�       �	�}!�Yc�A�&*

loss�JK<(�)       �	D"�Yc�A�&*

loss��<�q�J       �	�"�Yc�A�&*

loss�DM<�Ϳ       �	\U#�Yc�A�&*

loss1�<+4       �	b�#�Yc�A�&*

loss��<�"�       �	ݶ$�Yc�A�&*

loss�N$<����       �	UP%�Yc�A�&*

lossA?�<v��       �	�%�Yc�A�&*

lossг�<)�       �	<�&�Yc�A�&*

lossI��<T� �       �	Y�'�Yc�A�&*

loss��2<G���       �	
i(�Yc�A�&*

loss'�<0�       �	S)�Yc�A�&*

loss���<#g�N       �	m�)�Yc�A�&*

lossdfi=���       �	F�*�Yc�A�&*

loss3d�<�U-�       �	�o+�Yc�A�&*

loss���;u��       �	c,�Yc�A�&*

lossa�W=��x~       �	��,�Yc�A�&*

lossxq;N��       �	�l-�Yc�A�&*

loss|��<ި��       �	�.�Yc�A�&*

loss�>�;ٔ2�       �	~�.�Yc�A�&*

loss�"�<�h�<       �	�X/�Yc�A�&*

lossq�);6���       �	i0�Yc�A�&*

loss�h:a�q       �	1�Yc�A�&*

loss�7<�4�       �	ܹ1�Yc�A�&*

loss��D=��       �	,f2�Yc�A�&*

lossVg <�{f�       �	�	3�Yc�A�&*

loss\+�=��%       �	ګ3�Yc�A�&*

loss΄�;X>Pk       �	�`4�Yc�A�&*

loss �&<�@�O       �	�5�Yc�A�&*

loss(/;�`X       �	�5�Yc�A�&*

loss�
;�xM       �	g6�Yc�A�&*

loss��<��oi       �	�7�Yc�A�&*

loss��8=���@       �	�7�Yc�A�&*

loss<ē<�&�       �	�H8�Yc�A�&*

loss?J=2�ԯ       �	��8�Yc�A�&*

loss�(=���       �	9�Yc�A�&*

loss4VW=�-vU       �	)>:�Yc�A�&*

loss[_;�X�       �	w�:�Yc�A�&*

loss-��<�+�       �	er;�Yc�A�&*

loss�/=���       �	�<�Yc�A�&*

losst=�r�       �	,�<�Yc�A�&*

loss[P	=����       �	�S=�Yc�A�&*

lossa��=9M��       �	f�=�Yc�A�&*

losse%�;4���       �	��>�Yc�A�&*

loss��)<�r�       �	�[?�Yc�A�&*

loss�B�:V:��       �	��?�Yc�A�&*

loss���;����       �	��@�Yc�A�&*

loss�G�<�Z��       �	#A�Yc�A�&*

loss�$�;G�^        �	��A�Yc�A�&*

loss��;�w�@       �	c^B�Yc�A�&*

lossq��;����       �	�C�Yc�A�&*

loss�)�<�@       �	��C�Yc�A�&*

lossZ�3<ß%�       �	�FD�Yc�A�&*

lossV�<z:`       �	��D�Yc�A�&*

loss�<5ɗ�       �	H�E�Yc�A�&*

lossf��<�w�       �	G!F�Yc�A�&*

loss�N�<I�F       �	��F�Yc�A�&*

loss#��<�nZ\       �	�QG�Yc�A�&*

loss7�<��j�       �	��G�Yc�A�&*

loss�s�<�)�       �	��H�Yc�A�&*

loss��<�kV       �	@KI�Yc�A�&*

loss��V=�B�       �	�I�Yc�A�&*

loss�:�d�       �	�xJ�Yc�A�&*

loss��|<����       �	�K�Yc�A�&*

loss��*<O���       �	��K�Yc�A�&*

loss3�E< :8d       �	u�L�Yc�A�&*

loss:[�;��	       �	�7M�Yc�A�&*

loss@b�<=�L�       �	��M�Yc�A�&*

loss酆=.A�8       �	GwN�Yc�A�&*

lossE�z;�ג�       �	�O�Yc�A�&*

loss��<svb       �	G�O�Yc�A�&*

loss�o�;��p       �	�GP�Yc�A�&*

lossA��=�)�       �	��P�Yc�A�&*

loss�6�;�Xc�       �	xQ�Yc�A�'*

lossRX/<���A       �	-R�Yc�A�'*

loss�Ե:�L(       �	b�R�Yc�A�'*

lossN�<�2�*       �	X;S�Yc�A�'*

lossFx�: ��       �	x�S�Yc�A�'*

loss�K+=O��p       �	<hT�Yc�A�'*

lossg�<�t�       �	\�T�Yc�A�'*

loss܀<=ۨT       �		�U�Yc�A�'*

lossO;���       �	LV�Yc�A�'*

loss1��<�|       �	7�V�Yc�A�'*

loss8Zg<�,       �	){W�Yc�A�'*

loss?�<d�E'       �	�X�Yc�A�'*

loss�}�;S9f       �	H�X�Yc�A�'*

lossW�?<ث�.       �	=Y�Yc�A�'*

loss��<��L       �	�Y�Yc�A�'*

loss��=)��       �	0hZ�Yc�A�'*

loss���=���       �	[�Yc�A�'*

lossd�)=�SF=       �	Q�[�Yc�A�'*

lossd��<�Y��       �	�A\�Yc�A�'*

losst3�<��4       �	��\�Yc�A�'*

loss�K�;����       �	�p]�Yc�A�'*

loss��;z`�       �	j^�Yc�A�'*

loss��q<%r�       �	�^�Yc�A�'*

loss�;Y<z�d�       �	�J_�Yc�A�'*

loss�C"<�,�       �	 �_�Yc�A�'*

lossv��=��"�       �	|{`�Yc�A�'*

loss�]=_���       �	oa�Yc�A�'*

loss�8X=:J�       �	Φa�Yc�A�'*

loss;K=u��       �	�<b�Yc�A�'*

loss{=�Wo�       �	 �b�Yc�A�'*

lossl�;�z�       �	ic�Yc�A�'*

loss��$=g�J�       �	ed�Yc�A�'*

lossfQ�=���{       �	�d�Yc�A�'*

loss�:�<0�t[       �	4e�Yc�A�'*

loss�c�<JOV       �	a�e�Yc�A�'*

lossC�=<��<       �	yf�Yc�A�'*

loss���<�A�       �	Eg�Yc�A�'*

loss
��;2�T�       �	��g�Yc�A�'*

loss�v=�E<�       �	jLh�Yc�A�'*

loss3�;k�       �	\�h�Yc�A�'*

losscc<���       �	��i�Yc�A�'*

loss��B=�k       �	�Ij�Yc�A�'*

lossl��;�{@D       �	�j�Yc�A�'*

loss) �<�/�T       �	�k�Yc�A�'*

lossXz�<�2       �	R,l�Yc�A�'*

loss�}�<�.w       �	��l�Yc�A�'*

loss8X�<T�ҝ       �	sim�Yc�A�'*

loss,��<�qR�       �	Cn�Yc�A�'*

loss|{�<g�I�       �	ݙn�Yc�A�'*

loss�h�<��+�       �	0o�Yc�A�'*

lossH�9=�f�       �	��o�Yc�A�'*

loss��=^v�B       �	gbp�Yc�A�'*

loss���<v�r       �	>q�Yc�A�'*

loss��=ɷ��       �	E�q�Yc�A�'*

loss 0�<�郰       �	\8r�Yc�A�'*

loss��<�y3[       �	�r�Yc�A�'*

loss�J�;'Vd�       �	�hs�Yc�A�'*

loss�{x=ڊ[�       �	�s�Yc�A�'*

lossș�<
u��       �	0�t�Yc�A�'*

losspL=29G        �	VIu�Yc�A�'*

loss�9H>E�-�       �	�u�Yc�A�'*

loss�D�;�5��       �	=�v�Yc�A�'*

loss1�M;g�       �	#/w�Yc�A�'*

loss
��;��	;       �	��w�Yc�A�'*

loss�bA<@Ƴ�       �	q�x�Yc�A�'*

loss#`5=}!	�       �	<0y�Yc�A�'*

loss��Q<�ґ.       �	>�y�Yc�A�'*

loss��<��c       �	�tz�Yc�A�'*

loss��<D�`a       �	7{�Yc�A�'*

lossL�<�Oj�       �	|�{�Yc�A�'*

lossZ�;���_       �	)u|�Yc�A�'*

loss���=���A       �	�}�Yc�A�'*

loss�͒<���       �	��}�Yc�A�'*

lossL{=FLť       �	;U~�Yc�A�'*

loss�®<�k�       �	'�~�Yc�A�'*

loss�
k=�o�       �	X��Yc�A�'*

loss���<�i�       �	�<��Yc�A�'*

loss���<��0       �	�܀�Yc�A�'*

loss!oA<|�,Z       �	�x��Yc�A�'*

loss2�a<�hp�       �	v��Yc�A�'*

loss$�<��M       �	B���Yc�A�'*

loss|�<34�       �	�Z��Yc�A�'*

loss]a�<ܒ��       �	+���Yc�A�'*

loss�Ӡ=)}I=       �	����Yc�A�'*

loss_R�<�"z�       �	�8��Yc�A�'*

loss(�;X߾�       �	߅�Yc�A�'*

loss�q <���       �	6x��Yc�A�'*

loss���=�8�       �	N+��Yc�A�'*

loss@h�<~�/�       �	IՇ�Yc�A�'*

loss�3�<���H       �	�Ԉ�Yc�A�'*

loss!7�<���       �	����Yc�A�'*

loss�Ǆ;Pq�(       �	|a��Yc�A�'*

loss�T<���       �	�*��Yc�A�'*

loss��?=ԯ�       �	8��Yc�A�'*

loss�|B=i�\�       �	���Yc�A�'*

lossr�K=Zu�       �	�t��Yc�A�'*

loss2q�;MzS       �	?��Yc�A�'*

loss�U�<�$�       �	&Ȏ�Yc�A�'*

loss�<P��I       �	~n��Yc�A�'*

lossS!�:'Y:       �	��Yc�A�'*

loss��$=�У�       �	�Yc�A�'*

loss�xP<چB       �	0g��Yc�A�'*

loss1H;mQ-       �	���Yc�A�'*

lossw�:=�(�       �	����Yc�A�'*

loss���<1Ý&       �	�V��Yc�A�'*

loss�d=����       �	R�Yc�A�'*

loss} �<)8�       �	T���Yc�A�'*

loss�=y#e       �	n0��Yc�A�'*

lossV��=���       �	�ٕ�Yc�A�'*

lossHe�;`ʞ5       �	���Yc�A�'*

lossT| =�({       �	0��Yc�A�'*

loss�o-;}-,�       �	}ʘ�Yc�A�'*

lossMc�<��b�       �	�n��Yc�A�'*

loss�G�<��       �	h��Yc�A�'*

loss{�6>���s       �	񟚈Yc�A�'*

loss9W�:�^�A       �	<��Yc�A�'*

loss�#{<9ap       �	fڛ�Yc�A�'*

loss�W�<�f2s       �	2v��Yc�A�'*

loss���;"lvs       �	9~��Yc�A�'*

loss_�<�^�Y       �	%��Yc�A�'*

loss�e:=<V�       �	���Yc�A�'*

lossZ��=�۲       �	�v��Yc�A�'*

loss�Q�:X�Jo       �	���Yc�A�'*

lossH�+<mі       �	נ�Yc�A�'*

lossE)<�0�       �	6w��Yc�A�'*

loss���<e���       �	@��Yc�A�'*

losszԁ<쿦�       �	٢�Yc�A�'*

loss�-=�y��       �	w��Yc�A�'*

loss��;��O       �	\��Yc�A�(*

loss-�<4�       �	����Yc�A�(*

lossHS=�	r�       �	�U��Yc�A�(*

lossž�<��jS       �	��Yc�A�(*

loss��T;˵�       �	����Yc�A�(*

loss�K�<�ᇡ       �	�9��Yc�A�(*

loss���;�h�{       �	�ק�Yc�A�(*

lossZ��;^�D       �	����Yc�A�(*

loss�¢=LW>       �	B!��Yc�A�(*

lossrh<:Dwh       �	����Yc�A�(*

losss~�= �!^       �	V��Yc�A�(*

loss���;����       �	L���Yc�A�(*

lossϞC<��c�       �	ɏ��Yc�A�(*

loss�M%;	1�O       �	sL��Yc�A�(*

loss]	1<�i1       �	:鬈Yc�A�(*

loss�_�=�"M�       �	���Yc�A�(*

loss�`	<<�       �	�+��Yc�A�(*

loss,6;���       �	�Ʈ�Yc�A�(*

loss�y�<�G��       �	o���Yc�A�(*

lossSg�=����       �	��Yc�A�(*

loss��:EML       �	����Yc�A�(*

loss���<a|��       �	�w��Yc�A�(*

loss[�X<�);3       �	N*��Yc�A�(*

loss\J;XpSr       �	cҲ�Yc�A�(*

lossɒU=U${#       �	nm��Yc�A�(*

lossx��<Z�gg       �	�[��Yc�A�(*

loss�5=���       �	���Yc�A�(*

loss��;�X       �	���Yc�A�(*

lossV�=���X       �	�,��Yc�A�(*

loss�:4=ݨ`�       �	
ٶ�Yc�A�(*

loss�f�<e��       �	�q��Yc�A�(*

lossmK�:/A;�       �	���Yc�A�(*

lossf ;�V�       �	�ڸ�Yc�A�(*

loss�V�<�t@�       �	hx��Yc�A�(*

loss���=���       �	$��Yc�A�(*

loss�`=��D�       �	����Yc�A�(*

loss���<*x��       �	*8��Yc�A�(*

loss�t�;\k       �	��Yc�A�(*

loss��:=���Q       �	����Yc�A�(*

loss���;�m}f       �	@��Yc�A�(*

loss��<�_v-       �	2轈Yc�A�(*

loss���<d�w       �	b���Yc�A�(*

loss��Y<�Tc\       �	y!��Yc�A�(*

lossI��;�Xf�       �	-$��Yc�A�(*

loss\��<���K       �	����Yc�A�(*

loss�"�<�uW�       �	B`��Yc�A�(*

loss�ϵ<*��^       �	����Yc�A�(*

loss�`<F{Z       �	$�Yc�A�(*

loss���;�Ol       �	:ÈYc�A�(*

loss�+4<�b�n       �	��ÈYc�A�(*

loss��;�a��       �	mrĈYc�A�(*

loss2�=
�/�       �	ňYc�A�(*

loss��;���       �	�ňYc�A�(*

loss��X<t�+       �	VJƈYc�A�(*

loss=�]�X       �	;�ƈYc�A�(*

loss���==�g�       �	g~ǈYc�A�(*

loss�%x<:5v�       �	.ȈYc�A�(*

loss
�<3u0       �	��ȈYc�A�(*

loss{��=r�       �	��ɈYc�A�(*

lossn��<iؤ�       �	`�ʈYc�A�(*

lossނ=$5��       �	c	̈Yc�A�(*

loss���<fU�       �	M�̈Yc�A�(*

loss?�!<
�&�       �	R͈Yc�A�(*

loss���<�w�A       �	y;ΈYc�A�(*

loss��7<;N��       �	�ψYc�A�(*

loss�~ <��       �	#ЈYc�A�(*

lossF�=�F��       �	�шYc�A�(*

lossѽ =c�A�       �	R�шYc�A�(*

loss���<���!       �	[?҈Yc�A�(*

lossq4F<�b.       �	�]ӈYc�A�(*

lossȼZ;� �       �	�ӈYc�A�(*

lossc�<l��;       �	�ԈYc�A�(*

loss���=Z�       �	�GՈYc�A�(*

loss�'�;��,d       �	��ՈYc�A�(*

lossq�=�@	�       �	2�ֈYc�A�(*

lossW�T<��T�       �	�$׈Yc�A�(*

loss[��=��*J       �	�׈Yc�A�(*

lossd*�:�rV       �	.U؈Yc�A�(*

loss�`4<P*��       �	k�؈Yc�A�(*

loss���<	�w�       �	a�وYc�A�(*

lossn�O<��7       �	 ڈYc�A�(*

lossX�;�?��       �	x�ڈYc�A�(*

lossn�H<�Lb       �	�UۈYc�A�(*

loss��<�V�#       �	�ۈYc�A�(*

loss)|�<Ռ       �	Z�܈Yc�A�(*

loss($�<�֓       �	�-݈Yc�A�(*

lossq;�<Ü�       �	i�݈Yc�A�(*

lossߑ�<�g�@       �	W[ވYc�A�(*

lossD` <eAJw       �	D�ވYc�A�(*

loss���<B��       �	ō߈Yc�A�(*

loss���<���       �	$��Yc�A�(*

loss!�M<�|�       �	Ժ��Yc�A�(*

loss&r,=Z^       �	�l�Yc�A�(*

loss�(t<�!�       �	���Yc�A�(*

loss��<����       �	��Yc�A�(*

loss�=�e��       �	YP�Yc�A�(*

loss�X�<ܐ�E       �	��Yc�A�(*

lossC��;
�o       �	���Yc�A�(*

loss�Y;<T!�>       �	K!�Yc�A�(*

loss'�=X       �	U��Yc�A�(*

loss}�=д(�       �	�n�Yc�A�(*

loss��d<�v��       �	�*�Yc�A�(*

loss�}�=O�D       �	���Yc�A�(*

loss�@<J+'2       �	$~�Yc�A�(*

loss��&=ۡ͘       �	&�Yc�A�(*

loss��<�p�-       �	��Yc�A�(*

loss/2�<�1��       �	/��Yc�A�(*

loss=�<~T��       �	�=�Yc�A�(*

loss�.�=��7       �	���Yc�A�(*

loss&��<Xe�       �	Y��Yc�A�(*

loss�*�;�)��       �	�T�Yc�A�(*

lossT�8<�Zo       �	��Yc�A�(*

loss��<y���       �	r��Yc�A�(*

loss���<�=C4       �	PS��Yc�A�(*

loss
�d<��A       �	����Yc�A�(*

loss�&&=V�K       �	Ֆ�Yc�A�(*

loss�7�<ܳv       �	�0�Yc�A�(*

lossK;{���       �	(I�Yc�A�(*

loss�J�;0�L       �	���Yc�A�(*

lossO<��N�       �	���Yc�A�(*

loss��c<*�       �	�C��Yc�A�(*

loss�8�<1�/�       �	����Yc�A�(*

lossCQ/=S��       �	G���Yc�A�(*

loss���<&�Rd       �	����Yc�A�(*

loss?=DAI�       �	�t��Yc�A�(*

loss$<<В       �	���Yc�A�(*

loss-�7<�"��       �	����Yc�A�(*

loss�=P�l       �	Zc��Yc�A�(*

loss	�x;H1�q       �	����Yc�A�)*

lossXD<��Oo       �	����Yc�A�)*

loss�X�;       �	?��Yc�A�)*

lossw�<�;��       �	b���Yc�A�)*

loss�!;=��c�       �	Xu��Yc�A�)*

loss�� <d�z�       �	�
��Yc�A�)*

loss��=k�+       �	����Yc�A�)*

lossW@3=9!�i       �	?��Yc�A�)*

loss�h�=����       �	m �Yc�A�)*

loss�<�MTN       �	�� �Yc�A�)*

loss^ =��       �	�a�Yc�A�)*

loss2�2;+��       �	r��Yc�A�)*

loss��<~�~       �	��Yc�A�)*

lossW��<H�       �	W$�Yc�A�)*

loss�d;�\�       �	G��Yc�A�)*

lossҕ;=�9��       �	�p�Yc�A�)*

loss֏=�o       �	��Yc�A�)*

loss�%!=1R        �	��Yc�A�)*

loss�=����       �	�`�Yc�A�)*

lossS
=�Yo�       �	�
�Yc�A�)*

loss�T�<!ª       �	���Yc�A�)*

loss���;�h!       �	�E�Yc�A�)*

lossԪ:S�q       �	���Yc�A�)*

lossc��=�y�=       �	��	�Yc�A�)*

loss�6�;\�)       �	p�Yc�A�)*

loss���;ódw       �	�J�Yc�A�)*

loss��K=���       �	A��Yc�A�)*

lossD�</zĚ       �	S��Yc�A�)*

lossW+�;2�V�       �	�;�Yc�A�)*

lossa�:�Z�?       �	 ��Yc�A�)*

loss�IG<.v-W       �	5~�Yc�A�)*

loss�J�<lY�       �	G �Yc�A�)*

loss�Q=m��       �	��Yc�A�)*

loss$��;�_�m       �	�V�Yc�A�)*

loss3�v=�F�W       �	M��Yc�A�)*

loss�+<{]�V       �	��Yc�A�)*

lossaf�=�Y�       �	IH�Yc�A�)*

loss/rk=h       �	���Yc�A�)*

loss��/<m��       �	x�Yc�A�)*

loss�B�<1��3       �	#�Yc�A�)*

lossθ�=�W�       �	T��Yc�A�)*

loss�op<�wd       �	��Yc�A�)*

lossxW;=��?�       �	�7�Yc�A�)*

lossRM�;�Q�       �	g��Yc�A�)*

loss���<�w�[       �	�u�Yc�A�)*

loss�u<���A       �	�&�Yc�A�)*

loss�ٰ<���P       �	X��Yc�A�)*

loss�F�;�G�       �	|�Yc�A�)*

loss~�=����       �	�'�Yc�A�)*

lossh�<�w8�       �	%��Yc�A�)*

loss5<�$�*       �	Ot�Yc�A�)*

loss}�<e��n       �	u�Yc�A�)*

loss6'<a�V       �	��Yc�A�)*

loss�{U<zҿ       �	�W�Yc�A�)*

loss=,v<T<�       �	M��Yc�A�)*

loss!db<�I       �	͏�Yc�A�)*

loss�/�<\ڽm       �	J% �Yc�A�)*

lossB�=�)��       �	� �Yc�A�)*

lossH	%<w~�       �	DP!�Yc�A�)*

loss\��<hj:       �	N�!�Yc�A�)*

loss���;���1       �	z�"�Yc�A�)*

lossZ��<���5       �	�*#�Yc�A�)*

loss�<�0.�       �	��#�Yc�A�)*

loss�$�;��6�       �	�[$�Yc�A�)*

losst�<=���       �	E�$�Yc�A�)*

loss��;kO�1       �	+�%�Yc�A�)*

loss�/7;Ê�"       �	�!&�Yc�A�)*

lossK�=KL
       �	��&�Yc�A�)*

lossmy0<C�R�       �	<h'�Yc�A�)*

loss�j�<#�(f       �	q (�Yc�A�)*

lossf�;�\~N       �	��(�Yc�A�)*

lossA��< ��v       �	�()�Yc�A�)*

lossE
<<��d�       �	��)�Yc�A�)*

lossTW�:/.�y       �	)\*�Yc�A�)*

lossf�<(`�       �	V�*�Yc�A�)*

loss�|9�Ξ       �	r�+�Yc�A�)*

loss�=�<��A0       �	�d,�Yc�A�)*

loss��=�H       �	�-�Yc�A�)*

loss�>�<�f)       �	 �-�Yc�A�)*

loss���<�o�       �	�2.�Yc�A�)*

lossF`�<	:�       �	��.�Yc�A�)*

loss!�G=�X'       �	�]/�Yc�A�)*

loss��$<s���       �	Y0�Yc�A�)*

loss<Z;9�       �	{�0�Yc�A�)*

loss��=���       �	�T1�Yc�A�)*

lossSj:��4       �	��1�Yc�A�)*

lossQ�,;�h)       �	�2�Yc�A�)*

loss��;A��       �	((3�Yc�A�)*

loss�4�;��pk       �	��3�Yc�A�)*

loss���<4�q       �	_4�Yc�A�)*

loss�;k~C�       �	�4�Yc�A�)*

loss��:8l��       �	��5�Yc�A�)*

loss&9<���       �	�-6�Yc�A�)*

loss�BC;��43       �	}�6�Yc�A�)*

loss��:�$��       �	G�7�Yc�A�)*

loss�G<��*       �	�c8�Yc�A�)*

loss���<n٤�       �	��8�Yc�A�)*

loss�	�<5�#       �	S�9�Yc�A�)*

loss+	<�рE       �	�+:�Yc�A�)*

loss�Bv9���       �	ӥ;�Yc�A�)*

loss_��;LZw       �	�F<�Yc�A�)*

loss�=���L       �	��<�Yc�A�)*

loss�{�9�=�       �	�=�Yc�A�)*

loss���>eVm�       �	rj>�Yc�A�)*

lossmW�<�o�       �	�?�Yc�A�)*

lossW 2=��b&       �	��?�Yc�A�)*

loss�`<w��       �	$B@�Yc�A�)*

loss��W;���       �	�@�Yc�A�)*

loss��=���       �	k�A�Yc�A�)*

loss��<��X       �	-`B�Yc�A�)*

loss{4];�z�       �	P�B�Yc�A�)*

lossqq�;���	       �	�C�Yc�A�)*

loss7��<c���       �	�ED�Yc�A�)*

loss�!=�M�       �	��D�Yc�A�)*

loss���<�k+�       �	�E�Yc�A�)*

loss)�<%S�       �	k*F�Yc�A�)*

lossv�<y�И       �	!�F�Yc�A�)*

lossNB�<����       �	�eG�Yc�A�)*

loss��f;��\�       �	H�Yc�A�)*

lossvV=@���       �	��H�Yc�A�)*

loss��=e���       �	JDI�Yc�A�)*

loss߿4</D��       �	��J�Yc�A�)*

lossQ�!;*Di�       �	BK�Yc�A�)*

loss���=�k�       �	|�L�Yc�A�)*

loss0�<
'6�       �	EM�Yc�A�)*

loss5<���Y       �	%�M�Yc�A�)*

loss���:W��       �	P�N�Yc�A�)*

loss4<����       �	YPO�Yc�A�)*

loss��<��}L       �	%P�Yc�A�**

loss;��<�o�K       �	�P�Yc�A�**

loss�r�<=�       �	 �Q�Yc�A�**

loss@�J=��5       �	zSR�Yc�A�**

loss�ux<�7�       �	��R�Yc�A�**

lossx��<�'�       �	��S�Yc�A�**

loss%Q�=L���       �	�<T�Yc�A�**

losse(;4Mc�       �	t�T�Yc�A�**

loss���<���       �	kU�Yc�A�**

lossX׆<-��o       �	2 V�Yc�A�**

loss��9;�FN�       �	��V�Yc�A�**

loss��(<Ƥ�X       �	@/W�Yc�A�**

loss):=��       �	��W�Yc�A�**

loss�<���       �	�[X�Yc�A�**

loss<�;ʻ�g       �	$�X�Yc�A�**

loss#ݤ;]�       �	<�Y�Yc�A�**

loss)K�<*��       �	�Z�Yc�A�**

loss�>�<Ͻ�]       �	l�Z�Yc�A�**

lossj29;�.��       �	�L[�Yc�A�**

loss���<�"       �	q�[�Yc�A�**

loss#"z=��l       �	%w\�Yc�A�**

loss`�<��@�       �	JD]�Yc�A�**

loss&I�<}���       �	H�]�Yc�A�**

loss 4�;=Q�^       �	Y�^�Yc�A�**

lossl =N���       �	 _�Yc�A�**

loss�Bw<��Y5       �	yyw�Yc�A�**

loss�i=���       �	#x�Yc�A�**

loss�I=9�Cp       �	P�x�Yc�A�**

loss��g=F�(�       �	`�y�Yc�A�**

loss���<�dI       �	1�z�Yc�A�**

loss�G=��`       �	n{�Yc�A�**

loss4*�<�Y'�       �	�2|�Yc�A�**

loss�]�<p㽕       �	��|�Yc�A�**

loss�8�=�]5m       �	�i}�Yc�A�**

loss��</�       �	}�~�Yc�A�**

losss`�;�A��       �	q��Yc�A�**

loss�*<}$<�       �	�%��Yc�A�**

loss�eN<��{       �	eǀ�Yc�A�**

loss�S(<��z(       �	�`��Yc�A�**

loss���<祖       �	/���Yc�A�**

lossQ�0="A�       �	М��Yc�A�**

loss_�a;�<�       �	�?��Yc�A�**

loss=�;nd�Y       �	�탉Yc�A�**

loss�a�<���       �	z���Yc�A�**

loss)��<�f��       �	�$��Yc�A�**

loss���;"�[�       �	�х�Yc�A�**

loss΍C=�>w�       �	nh��Yc�A�**

loss�R;��]Z       �	r���Yc�A�**

loss84�=[m2�       �	����Yc�A�**

loss��<�Q�       �	,*��Yc�A�**

loss暰<�6�>       �	̈�Yc�A�**

loss!s�<\~�       �	-_��Yc�A�**

loss�?�;׿�       �	G��Yc�A�**

loss1�x=���M       �	�C��Yc�A�**

loss%�3<c��       �	n���Yc�A�**

loss�s<�ܙd       �	�5��Yc�A�**

loss�f<�]q       �	Y��Yc�A�**

loss!��<���A       �	�s��Yc�A�**

loss"C�=�j)       �	��Yc�A�**

loss<��ǿ       �	1���Yc�A�**

loss<�<��7�       �	�P��Yc�A�**

loss���;�R6	       �	���Yc�A�**

loss�D%=���       �	����Yc�A�**

loss���<Aǳ       �	�z��Yc�A�**

lossO�=]"E�       �	���Yc�A�**

loss�1=J:"|       �	�Ɣ�Yc�A�**

loss��:<����       �	4f��Yc�A�**

loss�Ɇ<u}��       �	��Yc�A�**

loss|��<�;X       �	ܺ��Yc�A�**

lossؚ<K!       �	�Z��Yc�A�**

loss�Ld=���        �	
.��Yc�A�**

loss��<�iM�       �	s��Yc�A�**

loss�<�<I�Ȫ       �	K噉Yc�A�**

loss{G+<4���       �	����Yc�A�**

lossL�;m\�       �	j0��Yc�A�**

loss5�;�:��       �	�˛�Yc�A�**

loss\3�<rr}�       �	˽��Yc�A�**

loss�<�|�       �	�b��Yc�A�**

losss��=8���       �	���Yc�A�**

loss�.�;W	|M       �	.���Yc�A�**

loss <	���       �	(E��Yc�A�**

loss�z	;G�D�       �	\埉Yc�A�**

lossD�J:��)n       �	�۠�Yc�A�**

loss`��;�5`�       �	lw��Yc�A�**

lossC�=�,|�       �	�!��Yc�A�**

lossr��=L��P       �	ܽ��Yc�A�**

loss1��<���       �	_��Yc�A�**

loss%�)<�.5L       �	����Yc�A�**

loss�0�<�`	�       �	c���Yc�A�**

losssR�;{K�2       �	��Yc�A�**

loss�K�;u@]       �	{���Yc�A�**

loss-�<_��8       �	�:��Yc�A�**

loss���;�ϝ�       �	(ҧ�Yc�A�**

loss�0�=�-�h       �	z��Yc�A�**

loss2��;?��f       �	
��Yc�A�**

lossNi<揞�       �	����Yc�A�**

loss��8<����       �	F?��Yc�A�**

loss �r=@~       �	�Ӫ�Yc�A�**

loss]n�;JyT�       �	�h��Yc�A�**

loss�Q[;'�       �	���Yc�A�**

lossX[�:��E       �	�٬�Yc�A�**

loss�<s��       �	�n��Yc�A�**

loss�,�;͔[�       �	���Yc�A�**

loss��&<��~       �	Y���Yc�A�**

loss�1=��J<       �	�I��Yc�A�**

loss
L�=,r��       �	䯉Yc�A�**

loss�r�<f���       �	����Yc�A�**

loss7��<=E       �	���Yc�A�**

lossE�0;�R�0       �	L8��Yc�A�**

loss�F)=��k       �	�ײ�Yc�A�**

loss�:�;Y�       �		��Yc�A�**

loss
�A<�b�d       �	���Yc�A�**

loss�;p<EY�       �	�N��Yc�A�**

loss���;H��       �	���Yc�A�**

losst+�:���X       �	q���Yc�A�**

loss _<���       �	�W��Yc�A�**

lossd��<�2�/       �	a���Yc�A�**

loss�
7<xO1�       �	ᙸ�Yc�A�**

loss*��;�4&       �	�E��Yc�A�**

lossL�<a��k       �	m㹉Yc�A�**

lossә�;�M:�       �	����Yc�A�**

loss��<Rc[W       �	�#��Yc�A�**

loss4�;"���       �	���Yc�A�**

loss/$�=N1��       �	W��Yc�A�**

losssٛ<e(��       �	k�Yc�A�**

loss���<���O       �	����Yc�A�**

loss�UB;��i       �	(��Yc�A�**

lossh�=��>�       �	���Yc�A�**

loss�o<T۬�       �	h[��Yc�A�+*

lossO
�;�/Wb       �	���Yc�A�+*

loss[-�<i :       �	���Yc�A�+*

loss�YJ;s6�       �	���Yc�A�+*

losso{}<���       �	�Yc�A�+*

lossO��=�Lo�       �	m ÉYc�A�+*

loss=ұ<�F��       �	t�ÉYc�A�+*

loss��<'�!�       �	e�ĉYc�A�+*

loss��=s��       �	�ƉYc�A�+*

lossn�=�7z;       �	N�ƉYc�A�+*

loss�w;�|\�       �	K>ǉYc�A�+*

loss2@�; 25       �	��ǉYc�A�+*

loss��;1P       �	lȉYc�A�+*

loss��;�7J[       �	yɉYc�A�+*

lossz�<�Gi       �	ÞɉYc�A�+*

loss{o^<�2��       �	a�ʉYc�A�+*

loss�:����       �	lzˉYc�A�+*

loss�:;��1�       �	^̉Yc�A�+*

loss�{�<'h�       �	�2͉Yc�A�+*

loss� <>B�s       �	l�͉Yc�A�+*

lossJc4=e�@�       �	q�ΉYc�A�+*

loss�E&=�c�y       �	%!ωYc�A�+*

lossFۓ<�@�t       �	>�ωYc�A�+*

loss��;��^�       �	&RЉYc�A�+*

lossq:<V0       �	��ЉYc�A�+*

loss�>�1�       �	�{щYc�A�+*

loss���<k�mD       �	�҉Yc�A�+*

loss�,�<�bg�       �	e�҉Yc�A�+*

lossM�)<�:��       �	�=ӉYc�A�+*

loss���<��p3       �	��ӉYc�A�+*

loss/�[<H=       �	
jԉYc�A�+*

loss�ߜ<� *       �	ՉYc�A�+*

loss�.:<15$       �	ӢՉYc�A�+*

loss���<$}F�       �	�;։Yc�A�+*

loss�<<���       �	�։Yc�A�+*

lossa�R;�Њ       �	�y׉Yc�A�+*

loss
��:��       �	G؉Yc�A�+*

loss�>O;�Y�m       �	�؉Yc�A�+*

loss��<3%˸       �	�vىYc�A�+*

loss�ss<���       �	�	ډYc�A�+*

loss�:=-��       �	��ډYc�A�+*

loss���<��1�       �	v8ۉYc�A�+*

loss��5=���&       �	�ۉYc�A�+*

loss�b�<�6d       �		m܉Yc�A�+*

loss�9<�(0-       �	� ݉Yc�A�+*

loss�8�;7�       �	1�݉Yc�A�+*

loss�2|<�k8�       �	�މYc�A�+*

loss�<�`�       �	M�߉Yc�A�+*

loss��>=5�^       �	L��Yc�A�+*

loss��t<�A��       �	%���Yc�A�+*

loss��<H)Si       �	�C�Yc�A�+*

loss �G=C��@       �	�$�Yc�A�+*

loss���9	a�       �	�'�Yc�A�+*

loss�v<Vn��       �	���Yc�A�+*

loss-Y�<��9       �	�_�Yc�A�+*

loss�UU=\��       �	E��Yc�A�+*

lossLV <#r�       �	��Yc�A�+*

lossn�f<Ӻ�       �	�)�Yc�A�+*

loss_w�;�%��       �	й�Yc�A�+*

loss���<끥�       �	�P�Yc�A�+*

lossXi�:��Ї       �	d��Yc�A�+*

loss(�!<�|��       �	��Yc�A�+*

lossD7%<�>�       �	�[�Yc�A�+*

loss[y�;�M9�       �	/4�Yc�A�+*

loss�)=����       �	���Yc�A�+*

loss�K�:�:        �	F[�Yc�A�+*

lossq!q<%       �	���Yc�A�+*

lossE�N=&?�!       �	���Yc�A�+*

loss� r<�qk�       �	iW�Yc�A�+*

loss�0�<ꓽB       �	���Yc�A�+*

loss���<J�B       �	���Yc�A�+*

loss�"�:t�M       �	���Yc�A�+*

losse�<C�dG       �	6���Yc�A�+*

lossJ�
=�&0       �	
1�Yc�A�+*

loss ��;T_�Q       �	R��Yc�A�+*

loss��<R�04       �	���Yc�A�+*

lossC�"<��"       �	a�Yc�A�+*

loss�}!:bf7=       �	%<��Yc�A�+*

loss*�<�'       �	0/��Yc�A�+*

loss��;�d�&       �	4���Yc�A�+*

lossW�<z��       �	�~��Yc�A�+*

loss��v;�愵       �	w��Yc�A�+*

loss��2;���       �	����Yc�A�+*

loss 3�<�_��       �	`?��Yc�A�+*

loss!�H=z�!3       �	���Yc�A�+*

loss]�<3ÄX       �	;q��Yc�A�+*

loss_m=]���       �	��Yc�A�+*

loss�q'=y�1       �	����Yc�A�+*

lossI�#=��X7       �	%?��Yc�A�+*

loss|wO=U��       �	����Yc�A�+*

loss��;���\       �	���Yc�A�+*

loss�;��	�       �	O��Yc�A�+*

loss�΄:'�q       �	���Yc�A�+*

loss&L�:T[I�       �	iW��Yc�A�+*

loss��=|��       �	p���Yc�A�+*

lossήP<\@VB       �	ǀ �Yc�A�+*

loss���;�Gq�       �	�$�Yc�A�+*

lossT��=B��       �	���Yc�A�+*

loss�j(;O�       �	�X�Yc�A�+*

lossRW=���r       �	���Yc�A�+*

loss���;�
�       �	���Yc�A�+*

loss��;<�Zi(       �	-�Yc�A�+*

loss�S;|p�       �	���Yc�A�+*

losshI=���       �	x`�Yc�A�+*

loss*��;���F       �	@��Yc�A�+*

loss�i|=x^��       �	���Yc�A�+*

lossݰ =+<7z       �	�*�Yc�A�+*

loss���<0]�       �	a��Yc�A�+*

lossE�=��<�       �	�Z�Yc�A�+*

loss�$�;z�`7       �	H��Yc�A�+*

loss�ß<�lo�       �	�	�Yc�A�+*

loss,�<�O�       �	�5
�Yc�A�+*

losso`:<���       �	��
�Yc�A�+*

loss��=*�       �	Hm�Yc�A�+*

lossQ�C;���       �	��Yc�A�+*

loss� =�a��       �	���Yc�A�+*

loss�(�<j�b�       �	�C�Yc�A�+*

lossG�<�r�8       �	H��Yc�A�+*

loss��;���       �	dy�Yc�A�+*

loss��;KȤY       �	��Yc�A�+*

lossz<_?Ӕ       �	���Yc�A�+*

lossxo]<�"�v       �	�@�Yc�A�+*

loss߾<���       �	���Yc�A�+*

lossʙ�;�ףZ       �	���Yc�A�+*

loss�9#<��)�       �	�7�Yc�A�+*

loss��@=��J       �	)��Yc�A�+*

loss���<?��v       �	ca�Yc�A�+*

loss�v,<�"j       �	� �Yc�A�+*

loss��:c��       �	��Yc�A�,*

loss<j�<]#�       �	G�Yc�A�,*

loss�R <s0�       �	��Yc�A�,*

loss��=4�O�       �	[{�Yc�A�,*

loss_j'=g�7       �	M�Yc�A�,*

loss��<����       �	���Yc�A�,*

loss!@�<�y��       �	�@�Yc�A�,*

loss���;_şm       �	E��Yc�A�,*

loss�U=�w��       �	-|�Yc�A�,*

lossL$=��dR       �	��Yc�A�,*

loss̯f<�s��       �	��Yc�A�,*

loss��<�3�q       �	J��Yc�A�,*

loss�-u<�A�w       �	40�Yc�A�,*

loss��=�Ά       �	:��Yc�A�,*

loss#�"<y��L       �	l_�Yc�A�,*

loss�9=���       �	/��Yc�A�,*

lossJ�<��       �	}��Yc�A�,*

loss�#�;Q�;�       �	�+�Yc�A�,*

loss��<S��       �	f��Yc�A�,*

loss6�;g�e       �	qU �Yc�A�,*

lossJ�<�"	�       �	�� �Yc�A�,*

loss<;��8       �	�!�Yc�A�,*

loss,�;vM5       �	+�"�Yc�A�,*

loss�Ft<E��s       �	�#�Yc�A�,*

loss���;��       �	�#�Yc�A�,*

loss��<1@8       �	�T$�Yc�A�,*

loss��<��c       �	G�$�Yc�A�,*

loss�=E�)�       �	��%�Yc�A�,*

loss.�<My+�       �	v&�Yc�A�,*

loss�;���o       �	��&�Yc�A�,*

loss,*=_W�       �	�R'�Yc�A�,*

loss���;8�H�       �	�'�Yc�A�,*

lossdB�<p�@       �	�(�Yc�A�,*

lossa)O<n�       �	<)�Yc�A�,*

loss�
:�JpL       �	a�)�Yc�A�,*

loss�?�<+5�       �	6?*�Yc�A�,*

loss��g<i�c�       �	9�*�Yc�A�,*

loss}�==b�E�       �	��+�Yc�A�,*

loss��u<dR       �	
L,�Yc�A�,*

loss�`�; 1=w       �	 �,�Yc�A�,*

loss:9<��%       �	��-�Yc�A�,*

lossH��;a�:E       �	.�Yc�A�,*

loss�B=N�?�       �	�.�Yc�A�,*

loss�ٷ=�S�       �	�=0�Yc�A�,*

loss�X�<֑��       �	g�0�Yc�A�,*

loss��s:u�L�       �	�1�Yc�A�,*

loss'�;�-O       �	32�Yc�A�,*

lossSy�;j�       �	��2�Yc�A�,*

lossnc�;	��"       �	k4�Yc�A�,*

loss -�9�K�       �	u�4�Yc�A�,*

lossM_�:��       �	�L5�Yc�A�,*

loss��; W�       �	��5�Yc�A�,*

loss�3=2�C�       �	-@7�Yc�A�,*

lossN��=����       �	��8�Yc�A�,*

loss���<�v       �	�N9�Yc�A�,*

loss#��<xB��       �	�9�Yc�A�,*

lossvG�;9f       �	��:�Yc�A�,*

lossڋ;�*�3       �	�<;�Yc�A�,*

lossi�;��!T       �	L�;�Yc�A�,*

losswp�:Nt��       �	��<�Yc�A�,*

loss洼:Z�&       �	)&=�Yc�A�,*

loss&� ;��~�       �	+�=�Yc�A�,*

loss��:3B:�       �	T�>�Yc�A�,*

loss���:#���       �	;�?�Yc�A�,*

loss3̭=4DK       �	,J@�Yc�A�,*

loss��<g��	       �	��@�Yc�A�,*

loss���;¢��       �	�A�Yc�A�,*

loss���<�'�        �	4�B�Yc�A�,*

loss�Ft;����       �	*9C�Yc�A�,*

lossg!<�_n?       �	D�Yc�A�,*

loss��=A<.3       �	j�D�Yc�A�,*

lossv�:`��y       �	PE�Yc�A�,*

loss㋄=��|�       �	��E�Yc�A�,*

loss(�<ܑ�h       �	5F�Yc�A�,*

loss���=�[C       �	�G�Yc�A�,*

lossŁ�;�$r�       �	5�G�Yc�A�,*

lossR��;gwHJ       �	AJH�Yc�A�,*

loss!�%;S6u       �	q�H�Yc�A�,*

loss�)V=���       �	��I�Yc�A�,*

losse�-<࢙5       �	^LJ�Yc�A�,*

loss��<�t�       �	�+K�Yc�A�,*

lossܺ<*ۇ5       �	��K�Yc�A�,*

loss�=J빾       �	�M�Yc�A�,*

loss.��;���       �	=�M�Yc�A�,*

loss4ׄ<;�       �	��N�Yc�A�,*

loss��:��E�       �	�2O�Yc�A�,*

loss���:�_�q       �	��O�Yc�A�,*

loss��<�,i�       �	��P�Yc�A�,*

loss��<:F�       �	�4Q�Yc�A�,*

loss.J�<���r       �	��Q�Yc�A�,*

loss�=�q7       �	�sR�Yc�A�,*

loss���<Qݬ�       �	�S�Yc�A�,*

loss|��: 2�       �	��S�Yc�A�,*

loss�)<VZ       �	��T�Yc�A�,*

loss2F�=-��h       �	�3U�Yc�A�,*

loss��&=�i        �	�
V�Yc�A�,*

loss�.p;8�        �	oIW�Yc�A�,*

loss̂�<Bl-       �	&�W�Yc�A�,*

loss��=���       �	��X�Yc�A�,*

lossv&i;��%�       �	D�Y�Yc�A�,*

loss�P_<����       �	�\Z�Yc�A�,*

loss��Q;�F�O       �	y;[�Yc�A�,*

lossjOx;!j<       �	@�[�Yc�A�,*

loss�,�<�I�       �	#�\�Yc�A�,*

loss[�=�3�       �	]�Yc�A�,*

loss(f�<ܷ��       �	^�]�Yc�A�,*

lossh�:����       �	�U^�Yc�A�,*

loss�g=#\�       �	�_�Yc�A�,*

loss�4;��|       �	��_�Yc�A�,*

loss�8�;���       �	�[`�Yc�A�,*

loss�W;��(j       �	��`�Yc�A�,*

loss�e�;`�D       �	z�a�Yc�A�,*

lossU;�BZ{       �	�Eb�Yc�A�,*

lossǲ=M�5�       �	��b�Yc�A�,*

loss�^�<q��"       �	7�c�Yc�A�,*

loss�;�;�       �	Md�Yc�A�,*

loss���<Ϡ�       �	h�d�Yc�A�,*

loss�[Q=	��       �	q�e�Yc�A�,*

loss:�I=>�{       �	�Ef�Yc�A�,*

loss�K=$,#�       �	��f�Yc�A�,*

loss	�Q;�Kn#       �	�{g�Yc�A�,*

loss��==d�       �	�"h�Yc�A�,*

losse�<�Bv�       �	��h�Yc�A�,*

loss�c�;���#       �	�hi�Yc�A�,*

loss�<�GN�       �	*j�Yc�A�,*

lossQ�d<���0       �	Q�j�Yc�A�,*

loss�M=[M       �	`k�Yc�A�,*

loss�`<c�co       �	� l�Yc�A�,*

lossMM_<n,�?       �	�l�Yc�A�-*

loss�F	=��':       �	Cm�Yc�A�-*

lossLe�<
i��       �	On�Yc�A�-*

loss�y�;���       �	��n�Yc�A�-*

lossα=���       �	�o�Yc�A�-*

loss��Y=
v�F       �	$Bp�Yc�A�-*

loss���<?/EW       �	_�p�Yc�A�-*

lossuh�=���&       �	�q�Yc�A�-*

loss�F�;��js       �	�4r�Yc�A�-*

loss���=���       �	��r�Yc�A�-*

lossm6=<y�O�       �	_t�Yc�A�-*

loss#܋=(N       �	��t�Yc�A�-*

lossV";�<`       �	��u�Yc�A�-*

loss��/<Uj_{       �	EKv�Yc�A�-*

loss}�<6��S       �	d�v�Yc�A�-*

loss�#Z<(S�       �	D�w�Yc�A�-*

lossvM<�)       �	W&x�Yc�A�-*

loss�<\}��       �	!�x�Yc�A�-*

loss��<` [6       �	`�y�Yc�A�-*

loss�&�<�\�b       �	q<z�Yc�A�-*

loss㉮=�K�x       �	��z�Yc�A�-*

loss%�*<k�        �	�x{�Yc�A�-*

lossew�<7B�       �	�|�Yc�A�-*

loss�}�=Uzq�       �	�|�Yc�A�-*

loss;�<\�e       �	x}�Yc�A�-*

loss���:$�+'       �	�~�Yc�A�-*

lossD�J=IvWm       �	�,�Yc�A�-*

lossj��<��0       �	���Yc�A�-*

loss�=�T��       �	_���Yc�A�-*

loss��;G�>�       �	�S��Yc�A�-*

lossT�; z�       �	�끊Yc�A�-*

lossdH\<��w�       �	�Ă�Yc�A�-*

loss^�<�G'       �	�`��Yc�A�-*

loss�X	<��6�       �	����Yc�A�-*

loss�<�q�       �	Ԙ��Yc�A�-*

loss��b=S �       �	�1��Yc�A�-*

losswc=�<�       �	�ʅ�Yc�A�-*

loss��:����       �	d��Yc�A�-*

loss3��;*��       �	u��Yc�A�-*

loss�/<���       �	]���Yc�A�-*

loss��<L<~�       �	4e��Yc�A�-*

loss֤�;J{��       �	K#��Yc�A�-*

loss�! <0Y�<       �	Y���Yc�A�-*

lossQ`�=���       �	�V��Yc�A�-*

lossRT<ЗE�       �	����Yc�A�-*

loss�#=H�-�       �	���Yc�A�-*

losss}�;KA�       �	j1��Yc�A�-*

lossށ<.p       �	JҌ�Yc�A�-*

lossd,�==�/S       �	+m��Yc�A�-*

loss���<�[--       �	&���Yc�A�-*

lossn��=!w       �	G��Yc�A�-*

loss��=�1u�       �	i㏊Yc�A�-*

loss� A=�
v�       �	����Yc�A�-*

loss�F*=�ʡ�       �	8-��Yc�A�-*

lossTF�<�tr�       �	iƑ�Yc�A�-*

loss<=�}��       �	[a��Yc�A�-*

loss�b=XQ�       �	�`��Yc�A�-*

lossPE<Qm�       �	����Yc�A�-*

loss��<p���       �	ǝ��Yc�A�-*

lossEL�;!��o       �	�F��Yc�A�-*

loss�d!=tɇ       �	�않Yc�A�-*

loss�<�t       �	����Yc�A�-*

loss��%=�.Ϡ       �	H6��Yc�A�-*

lossF/<��	}       �	����Yc�A�-*

loss���=y�ES       �	a���Yc�A�-*

loss}��<�M��       �	�M��Yc�A�-*

lossa�K=�gr       �	2虊Yc�A�-*

loss'�<�g}:       �	�ؚ�Yc�A�-*

loss�jY=��       �	7ߛ�Yc�A�-*

loss_ʵ<�QV�       �	o~��Yc�A�-*

loss,��:�?�       �	j��Yc�A�-*

loss��!<J6A�       �	����Yc�A�-*

loss��<�і�       �	�I��Yc�A�-*

loss��;�L�       �	�Yc�A�-*

lossNm=�[�       �	Ҍ��Yc�A�-*

loss�h=%e�H       �	\;��Yc�A�-*

lossK�=�r1       �	�٠�Yc�A�-*

loss�Ǆ<�04,       �	\v��Yc�A�-*

loss���;�f�h       �	���Yc�A�-*

loss��<�ei       �	좊Yc�A�-*

loss���<DM��       �	6���Yc�A�-*

loss��=�b�       �	�/��Yc�A�-*

lossOʌ=N#j       �	Ϥ�Yc�A�-*

loss�<9�0�       �	Ww��Yc�A�-*

loss�4�=Zn�^       �	���Yc�A�-*

loss�;�<��
�       �	콦�Yc�A�-*

loss�¦<���:       �	�f��Yc�A�-*

loss��P=I=Qy       �	
��Yc�A�-*

loss6�<��       �	����Yc�A�-*

loss̚`<=��       �	�X��Yc�A�-*

loss 6J<�$E$       �	����Yc�A�-*

lossM�<_!��       �	+���Yc�A�-*

loss�8r<Q@a�       �	�T��Yc�A�-*

loss�;��<       �	����Yc�A�-*

loss_��<�4       �	���Yc�A�-*

loss��5=�&�       �	�b��Yc�A�-*

loss��5=���       �	@��Yc�A�-*

lossVE�<�f�       �	c���Yc�A�-*

loss�*8<��ٲ       �	a��Yc�A�-*

loss�=Z��	       �	���Yc�A�-*

loss�g<�D%�       �	Υ��Yc�A�-*

loss&:�;��       �	M��Yc�A�-*

loss���:��n       �	_ﱊYc�A�-*

lossi��:��_�       �	\��Yc�A�-*

loss�<=�ܜ�       �	�V��Yc�A�-*

lossv�;� R�       �	�
��Yc�A�-*

lossӖ4<���w       �	Sv��Yc�A�-*

loss��<��5       �	(-��Yc�A�-*

loss?��<�;�+       �	�;��Yc�A�-*

loss��=�]�       �	�n��Yc�A�-*

loss�{�;�Z�       �	+/��Yc�A�-*

loss���=��       �	���Yc�A�-*

lossC;�4s       �	ҩ��Yc�A�-*

loss*��=����       �	VF��Yc�A�-*

losscvJ=�X<       �	~⼊Yc�A�-*

loss7
�;��       �	����Yc�A�-*

loss߻W<�#�'       �	ղ��Yc�A�-*

loss�8m;{���       �	�տ�Yc�A�-*

loss���<���9       �	ap��Yc�A�-*

loss.�+<�C       �	��Yc�A�-*

loss`�=�z��       �	)���Yc�A�-*

loss���=
68A       �	0�Yc�A�-*

lossZWc=v�N�       �	5ÊYc�A�-*

loss�n�<���       �	��ÊYc�A�-*

lossZ��;37��       �	iĊYc�A�-*

loss�^=���k       �	VŊYc�A�-*

loss�l;�9��       �	�ŊYc�A�-*

loss��n<��h�       �	�TƊYc�A�-*

lossd�=���       �	�ƊYc�A�.*

lossީ=kn�       �	�ǊYc�A�.*

loss�|=�̟�       �	t#ȊYc�A�.*

losst��;��Y�       �	~�ȊYc�A�.*

loss�Ū;O�
�       �	�bɊYc�A�.*

loss7�[;x̿a       �	�
ʊYc�A�.*

loss���<suS�       �	��ʊYc�A�.*

loss�#U=�O��       �	ۆˊYc�A�.*

loss�͊<�֥L       �	f�̊Yc�A�.*

lossZ̈<��f       �	
�͊Yc�A�.*

loss�a�;~�E9       �	�ΊYc�A�.*

lossi�</�        �	B	ϊYc�A�.*

loss���<����       �	ЊYc�A�.*

losswme<����       �	o�ЊYc�A�.*

loss���;�D}A       �	|�ъYc�A�.*

losss�^;)�Rr       �	÷ҊYc�A�.*

loss}��<���,       �	�lӊYc�A�.*

loss
iO=�5
       �	�	ԊYc�A�.*

loss8@<n���       �	K�ԊYc�A�.*

lossԜ�<��!�       �	��ՊYc�A�.*

loss1��<��       �	�{֊Yc�A�.*

loss��<���       �	'׊Yc�A�.*

lossV��<�p��       �	�I؊Yc�A�.*

loss
�<fB�       �	��؊Yc�A�.*

loss_�<n,�Y       �	��يYc�A�.*

loss��$<���       �	��ڊYc�A�.*

lossc�<
�0�       �	ƅۊYc�A�.*

lossd��<��r       �	�g܊Yc�A�.*

loss	�M=koBA       �	��܊Yc�A�.*

loss��<킹r       �	�:ފYc�A�.*

loss��;�Fj       �	sߊYc�A�.*

loss�i<J�       �	}��Yc�A�.*

loss�A=&��       �	�]�Yc�A�.*

loss�$C;
{vg       �	C�Yc�A�.*

lossĺp=B��	       �	��Yc�A�.*

loss�;�!��       �	A��Yc�A�.*

loss��D<a�P       �	�@�Yc�A�.*

loss�	<o�       �	�,�Yc�A�.*

loss�J�<��|#       �	���Yc�A�.*

loss�?�:��       �	���Yc�A�.*

loss@�=<�wDd       �	� �Yc�A�.*

loss(=S&�       �	ͫ�Yc�A�.*

loss���<9��       �	d�Yc�A�.*

loss+�=Z,	F       �	U�Yc�A�.*

loss�<�kǕ       �	���Yc�A�.*

loss׊�;u��       �	�*�Yc�A�.*

lossw%�:��       �	���Yc�A�.*

loss\W�;鰺�       �	
��Yc�A�.*

loss��5<ū�]       �	=�Yc�A�.*

loss�(<���}       �	���Yc�A�.*

loss8��;q�a�       �	�z�Yc�A�.*

lossbN�9��a�       �	���Yc�A�.*

loss�o�;؜g#       �	����Yc�A�.*

loss|ʒ9UŴ�       �	3P�Yc�A�.*

loss� e9���*       �	V��Yc�A�.*

loss�4:ӹ�       �	Ĕ�Yc�A�.*

loss���:�!�I       �	2�Yc�A�.*

loss�Ɇ;�_�N       �	���Yc�A�.*

loss�}�; %	�       �	�g�Yc�A�.*

lossh <���V       �	���Yc�A�.*

loss�C;��M       �	4���Yc�A�.*

lossd;�=p!O       �	>=��Yc�A�.*

loss�?	<�tI�       �	����Yc�A�.*

loss�ؘ=);��       �	����Yc�A�.*

lossk�=Y_�x       �	c%��Yc�A�.*

loss�q�=>?��       �	����Yc�A�.*

loss�OC<��%}       �	>^��Yc�A�.*

loss\O�<��t�       �	4���Yc�A�.*

loss���<3^^$       �	.��Yc�A�.*

lossJ��<Cj#       �	���Yc�A�.*

loss���<� �0       �	yW��Yc�A�.*

loss��-=�ɶ�       �	I���Yc�A�.*

loss�c�;2�       �	.���Yc�A�.*

lossZ_<P���       �	�D��Yc�A�.*

loss1��<ک�J       �	����Yc�A�.*

loss a�;L��
       �	�9 �Yc�A�.*

loss�?<\       �	�� �Yc�A�.*

loss1�"=��       �	�p�Yc�A�.*

loss��<̏&y       �	�Yc�A�.*

loss�&;ʛ�       �	���Yc�A�.*

loss_�s=�o       �	{I�Yc�A�.*

loss���<���       �	?��Yc�A�.*

loss�T�<��O�       �	���Yc�A�.*

loss��<���       �	�4�Yc�A�.*

loss��<Lu��       �	���Yc�A�.*

loss��:��j       �	���Yc�A�.*

lossmp�<��       �	��Yc�A�.*

loss��)<��g       �	���Yc�A�.*

loss���<"�<�       �	�N�Yc�A�.*

loss6�9=�%4       �	��Yc�A�.*

losst�`=F~.y       �	��	�Yc�A�.*

loss��<W;�
       �	` 
�Yc�A�.*

loss�7;=���]       �	f�
�Yc�A�.*

loss��<ɜ       �	�|�Yc�A�.*

losshŋ;/JC�       �	�V�Yc�A�.*

loss��<��q�       �	��Yc�A�.*

loss�;��;G       �	ܸ�Yc�A�.*

loss1J�:*��Z       �	�_�Yc�A�.*

loss��;^�5L       �	�C�Yc�A�.*

loss��7=���       �	Q/�Yc�A�.*

lossVAc=�c�T       �	���Yc�A�.*

loss��;���F       �	A�Yc�A�.*

loss��x<�+s       �	f��Yc�A�.*

loss	��<�Ov       �	���Yc�A�.*

loss�T<J�       �	t&�Yc�A�.*

lossd��<3�_�       �	���Yc�A�.*

loss|<�ʶ�       �	Ab�Yc�A�.*

loss�$"=ц��       �	g�Yc�A�.*

lossW�=.G,�       �	)��Yc�A�.*

loss%�u<�z�V       �	GY�Yc�A�.*

loss#a�<^0       �	Y��Yc�A�.*

loss�h�;a�Nb       �	ǝ�Yc�A�.*

loss��:<��MN       �	�A�Yc�A�.*

loss�c<�7�       �	��/�Yc�A�.*

lossl=dj�       �	.p0�Yc�A�.*

lossÄ=o�
       �	�1�Yc�A�.*

lossh�<֧]       �	�1�Yc�A�.*

lossL=�U       �	�D2�Yc�A�.*

loss.�W;H�R�       �	B3�Yc�A�.*

loss{<U�b�       �	Y�3�Yc�A�.*

loss���;x��        �	p>4�Yc�A�.*

loss�~�=�7S       �	��4�Yc�A�.*

lossT��:��'       �	��5�Yc�A�.*

loss�;f�x�       �	�66�Yc�A�.*

loss6�;郥�       �	1�6�Yc�A�.*

loss�J�<�ӽ       �	A}7�Yc�A�.*

loss@��<��l       �	��8�Yc�A�.*

loss�r=�I��       �	f19�Yc�A�.*

loss\:�= L��       �	��9�Yc�A�/*

loss\�;bVț       �	sf:�Yc�A�/*

loss��<Mϼ�       �	�1;�Yc�A�/*

loss$�Y<�n�       �	E�;�Yc�A�/*

loss�<��       �	cz<�Yc�A�/*

lossz�;F=�8       �	�.=�Yc�A�/*

loss�?�=xs�j       �	��=�Yc�A�/*

loss3׀;�C�       �	�l>�Yc�A�/*

loss���=�6��       �	.?�Yc�A�/*

losse�|=-�>       �	;�?�Yc�A�/*

lossNX�;��q\       �	g@�Yc�A�/*

lossc�<'>%       �	��@�Yc�A�/*

loss�~<���[       �	��A�Yc�A�/*

loss��c=!q�\       �	7UB�Yc�A�/*

loss��<�A}Y       �	�7C�Yc�A�/*

lossD�;i��4       �	�C�Yc�A�/*

lossDC�<���2       �	�yD�Yc�A�/*

loss�^|<� z#       �	�E�Yc�A�/*

loss2=����       �	�E�Yc�A�/*

losszO<�B��       �	.SF�Yc�A�/*

loss��/=9�w@       �	��F�Yc�A�/*

loss�<v?       �	��G�Yc�A�/*

loss�K<�(X�       �	�8H�Yc�A�/*

loss�f�<�V�       �	{�H�Yc�A�/*

loss�e=�ă�       �	vI�Yc�A�/*

loss:�<�(�       �	GJ�Yc�A�/*

loss�J�<�+�;       �	ϾJ�Yc�A�/*

loss,��;M[��       �	mYK�Yc�A�/*

loss��1<����       �	�K�Yc�A�/*

loss�<3m��       �	ϢL�Yc�A�/*

loss��(=�a�Q       �	@M�Yc�A�/*

loss���<�N�       �	��M�Yc�A�/*

loss���; OC�       �	G�N�Yc�A�/*

loss�۪<~��g       �	�.O�Yc�A�/*

loss<�3<��mT       �	��O�Yc�A�/*

loss�49:βN       �	�jP�Yc�A�/*

loss��=���       �	>Q�Yc�A�/*

lossvr(<���n       �	��Q�Yc�A�/*

loss�ί=<t�c       �	!<R�Yc�A�/*

loss(<�;ݱ��       �	��R�Yc�A�/*

loss�)5;���       �	�nS�Yc�A�/*

loss�ބ;�Ŭ       �	ST�Yc�A�/*

loss�L;C/�!       �	��T�Yc�A�/*

loss��;9�E\       �	�U�Yc�A�/*

loss��< ��       �	�3V�Yc�A�/*

loss��=5u�"       �	��V�Yc�A�/*

loss��<�}^�       �	�\W�Yc�A�/*

loss�\=��)4       �	I�W�Yc�A�/*

loss W=��       �	��X�Yc�A�/*

loss��=H�@       �	˃Y�Yc�A�/*

lossء�:�H'C       �	�Z�Yc�A�/*

losse<00��       �	��Z�Yc�A�/*

loss�YG<���       �	Di[�Yc�A�/*

loss�1=bn��       �	,\�Yc�A�/*

loss���<��z       �	��\�Yc�A�/*

loss�hN=�+�       �	�x]�Yc�A�/*

loss�1<�c��       �	!@^�Yc�A�/*

loss�u=����       �	��^�Yc�A�/*

loss,�I;3/R�       �	��_�Yc�A�/*

lossCB�;K2g       �	�`�Yc�A�/*

lossV��<��-�       �	��`�Yc�A�/*

loss /�;��{�       �	7Qa�Yc�A�/*

lossh_8=3� �       �	R�a�Yc�A�/*

loss���<�o       �	8�b�Yc�A�/*

loss��<|Y[       �	�Sc�Yc�A�/*

loss�=�;>d�       �	d�c�Yc�A�/*

lossu0�;�%�       �	�d�Yc�A�/*

loss:�$<�[��       �	e�Yc�A�/*

lossL	z;2K<�       �	��e�Yc�A�/*

lossE��;�d�       �	�Lf�Yc�A�/*

loss���;�D��       �	��f�Yc�A�/*

lossfo�<&^�       �	�~g�Yc�A�/*

loss7�*=tQ�R       �	� h�Yc�A�/*

loss�<��(       �	�h�Yc�A�/*

loss�L<T���       �	�ki�Yc�A�/*

loss�["<���l       �	*j�Yc�A�/*

loss@'�<�`!�       �	��j�Yc�A�/*

loss$}<^���       �	75k�Yc�A�/*

loss?
V=�+ �       �	��k�Yc�A�/*

lossO3�<*��k       �	N_l�Yc�A�/*

loss���<:�       �	�m�Yc�A�/*

loss��"=P�ׂ       �	]�m�Yc�A�/*

loss{��;<CT�       �	.Yn�Yc�A�/*

loss�='��?       �	��n�Yc�A�/*

lossVe�;��g�       �	��o�Yc�A�/*

lossHA�;����       �	U�p�Yc�A�/*

loss�c�<屓
       �	�Eq�Yc�A�/*

loss��;_�\K       �	��q�Yc�A�/*

loss�4f;�Z&        �	~r�Yc�A�/*

loss#у<��        �	�s�Yc�A�/*

loss���;���       �	Gt�Yc�A�/*

lossH�O<k�5       �	��t�Yc�A�/*

loss��X;�	v�       �	�Fu�Yc�A�/*

losss\=F�       �	X�u�Yc�A�/*

loss�`�<4z�-       �	|v�Yc�A�/*

loss�=r;J       �	�w�Yc�A�/*

lossʋ�<�,4�       �	�w�Yc�A�/*

loss��
=>l��       �	��x�Yc�A�/*

lossb>;����       �	�y�Yc�A�/*

lossJ@<�_U9       �	N+z�Yc�A�/*

loss��:�=>/       �	.�z�Yc�A�/*

lossߊ�;Qc}\       �	�{�Yc�A�/*

losso�)=�c��       �	o-|�Yc�A�/*

loss���<����       �	��|�Yc�A�/*

loss$�<4���       �	�[}�Yc�A�/*

loss��<�B��       �	�,~�Yc�A�/*

lossi�J<{��       �	i�~�Yc�A�/*

loss���<��c2       �	�_�Yc�A�/*

loss�h=<��g       �	I��Yc�A�/*

loss5�=���$       �	玀�Yc�A�/*

loss��;/^�       �	�#��Yc�A�/*

loss
F�<��t_       �	���Yc�A�/*

loss-�X=�a�'       �	_���Yc�A�/*

loss�
N=,�       �	1��Yc�A�/*

loss�<��Y       �	�;��Yc�A�/*

lossPa�<���       �	��Yc�A�/*

loss�=�(�       �	󐅋Yc�A�/*

lossMc=����       �	�'��Yc�A�/*

losshd=���       �	]���Yc�A�/*

loss$ <�O"J       �	U��Yc�A�/*

losss�<%�y�       �	=Yc�A�/*

loss�t�;7�ؓ       �	����Yc�A�/*

loss���;C��S       �	>%��Yc�A�/*

loss�+O<�7L2       �	Cǉ�Yc�A�/*

loss�t�<�/	T       �	�l��Yc�A�/*

loss"�</�       �	���Yc�A�/*

loss��%<��       �	s���Yc�A�/*

loss�� =ˡY�       �	�r��Yc�A�/*

loss�J�;�'�       �	0g��Yc�A�0*

losslx�<�Kn�       �	(��Yc�A�0*

loss��;�d(�       �	���Yc�A�0*

loss�U<�8�       �	�s��Yc�A�0*

loss(B;K5��       �	*V��Yc�A�0*

loss$�:r�       �	5��Yc�A�0*

lossi�c;:x4       �	WΑ�Yc�A�0*

loss�R<���       �	k���Yc�A�0*

lossSA=w��       �	[(��Yc�A�0*

loss��u=�Y��       �	=`��Yc�A�0*

loss���<_k       �	����Yc�A�0*

loss���<��VS       �	ڕ�Yc�A�0*

lossa�j<�Y��       �	�ɖ�Yc�A�0*

loss��{<�ԥ       �	����Yc�A�0*

loss�̊<���6       �	�o��Yc�A�0*

loss$<r�       �	i��Yc�A�0*

loss�1�<J�ρ       �	ke��Yc�A�0*

loss�i�<�v8�       �	H7��Yc�A�0*

lossuG�<���j       �	�n��Yc�A�0*

loss��A=�(v}       �	�J��Yc�A�0*

loss��::�ʠ.       �	���Yc�A�0*

loss%Ъ<֢$       �	���Yc�A�0*

loss�c<1���       �	����Yc�A�0*

loss�< �u3       �	�;��Yc�A�0*

loss�h]; �@       �	䡋Yc�A�0*

lossTƢ<1r�       �	����Yc�A�0*

lossVp�<��
�       �	�/��Yc�A�0*

loss�v�<���       �	|C��Yc�A�0*

loss�-�<z,       �	]n��Yc�A�0*

lossa�;�t��       �	P��Yc�A�0*

loss���;5Ϳ       �	��Yc�A�0*

losszS�<�Рi       �	ȴ��Yc�A�0*

loss"[:Õ|       �	狨�Yc�A�0*

loss�� <���4       �	#L��Yc�A�0*

loss�<L�       �	���Yc�A�0*

loss�
<��       �	qȪ�Yc�A�0*

loss�ac<�u�       �	G��Yc�A�0*

loss6�9�uwI       �	-ͬ�Yc�A�0*

loss�<!}@j       �	�`��Yc�A�0*

lossz~�<��Z�       �	[D��Yc�A�0*

loss�;F���       �	Zٮ�Yc�A�0*

lossH<�@�M       �	�o��Yc�A�0*

lossE�<��m�       �	J��Yc�A�0*

loss�GD=�M(       �		���Yc�A�0*

loss��;�i��       �	�2��Yc�A�0*

loss���:4�C       �	�ų�Yc�A�0*

loss��i<���       �	KY��Yc�A�0*

loss��f<,�bq       �	紋Yc�A�0*

loss�u�;�.��       �	l|��Yc�A�0*

loss��;1�:�       �	o��Yc�A�0*

loss���<$��       �	ܡ��Yc�A�0*

loss���9�7��       �	P8��Yc�A�0*

loss�OG<��W       �	渋Yc�A�0*

loss���;�#{       �	�|��Yc�A�0*

loss@�e;5���       �	���Yc�A�0*

loss�7;�s�       �	ũ��Yc�A�0*

losso�;-�l�       �	cC��Yc�A�0*

loss꽼:_�)-       �	�޻�Yc�A�0*

lossn�_;}BHf       �	f���Yc�A�0*

loss��;��̺       �	!��Yc�A�0*

loss�HN=���c       �	�Ľ�Yc�A�0*

loss�="��\       �	@P��Yc�A�0*

loss�<=�c��       �	{���Yc�A�0*

loss�Zn=�q�o       �	C���Yc�A�0*

loss���<�^;       �	AG��Yc�A�0*

loss$#h;9LE�       �	����Yc�A�0*

loss�_�<4L��       �	q�Yc�A�0*

lossV^�<a�r5       �	$EËYc�A�0*

loss;p+;z"a       �	��ËYc�A�0*

loss�m<hJ�       �	�~ċYc�A�0*

loss6�p;�%ty       �	�ŋYc�A�0*

loss���<�1�       �	O�ŋYc�A�0*

loss|��=�&��       �	�RƋYc�A�0*

loss{��=Mm�       �	��ƋYc�A�0*

loss
>"#a�       �	�~ǋYc�A�0*

loss���<�:       �	�ȋYc�A�0*

loss]}<��
�       �	ũȋYc�A�0*

lossA7=����       �	IMɋYc�A�0*

loss���;�PΘ       �	��ɋYc�A�0*

loss���<�l�-       �	{ʋYc�A�0*

loss]�J=��$`       �	{ˋYc�A�0*

lossz1�<�qW=       �	�ˋYc�A�0*

lossCq
<�A�       �	�̋Yc�A�0*

loss�g?=�4^.       �	X͋Yc�A�0*

loss���;y��x       �	�͋Yc�A�0*

loss�y�;T�R       �	ę΋Yc�A�0*

loss���<�{       �	n4ϋYc�A�0*

lossS�M<��       �	0�ϋYc�A�0*

loss�';���A       �	/kЋYc�A�0*

loss�x�=�F�G       �	6ыYc�A�0*

loss���;UJ*}       �	�ыYc�A�0*

lossl_]=� K�       �	�7ҋYc�A�0*

loss��=*P��       �	%�ҋYc�A�0*

loss�Ҥ;��9       �	�qӋYc�A�0*

loss���;�10�       �	�ԋYc�A�0*

lossŜ�<��v       �	ܜԋYc�A�0*

lossH��<��9       �	�3ՋYc�A�0*

loss1�<�2�M       �	��ՋYc�A�0*

loss�E'=3��       �	c֋Yc�A�0*

loss���=�BY       �	�֋Yc�A�0*

loss,�X<Y?�       �	|�׋Yc�A�0*

loss@��<��lP       �	�1؋Yc�A�0*

loss�O�;�d/       �	K�؋Yc�A�0*

loss��>;��L       �	F^ًYc�A�0*

lossl=*���       �	�ڋYc�A�0*

lossF/=_�"�       �	��ڋYc�A�0*

loss��;�/��       �	�TۋYc�A�0*

loss��D;'W_h       �	j�ۋYc�A�0*

loss�C=Y
@�       �	C�܋Yc�A�0*

loss��n<T�O       �	�)݋Yc�A�0*

loss9�;5�        �	��݋Yc�A�0*

lossla;����       �	aދYc�A�0*

lossݻ�;(�y       �	7�ދYc�A�0*

lossof�;���       �	��ߋYc�A�0*

loss�7;�gձ       �	�5��Yc�A�0*

loss�<�L       �	����Yc�A�0*

loss[.=�a��       �	���Yc�A�0*

loss�o<𿊰       �	�{�Yc�A�0*

loss�v�;:���       �	�&�Yc�A�0*

lossf�<��       �	4��Yc�A�0*

lossl)Z<ܲz       �	SY�Yc�A�0*

loss�=*Mk�       �	k��Yc�A�0*

loss��<�0��       �	���Yc�A�0*

loss�\5<y�\�       �	�#�Yc�A�0*

lossW�&=�]�       �	M��Yc�A�0*

loss��<k��z       �	ʌ�Yc�A�0*

loss�C<���k       �	�)�Yc�A�0*

lossq	�;Nd|�       �	���Yc�A�0*

loss�l�;V���       �	mr�Yc�A�1*

lossxպ;��5�       �	s�Yc�A�1*

loss�F; ��       �	��Yc�A�1*

loss�5�=��a�       �	Ih�Yc�A�1*

lossh<]脟       �	z �Yc�A�1*

loss<��:z��       �	���Yc�A�1*

loss_��;��b�       �	g�Yc�A�1*

loss�#�=K�{       �	�Yc�A�1*

loss;;=O��J       �	g��Yc�A�1*

loss��<�S        �	�8��Yc�A�1*

loss�;^V�A       �	EK�Yc�A�1*

loss�!�:�&��       �	��Yc�A�1*

lossll-<�V��       �	�~�Yc�A�1*

loss�Z2<]4V       �	��Yc�A�1*

loss���;�D�       �	���Yc�A�1*

lossI�i<��>�       �	�Y�Yc�A�1*

lossCQ�;?t�x       �	B��Yc�A�1*

loss�=��g       �	���Yc�A�1*

loss�3	;�c�       �	��Yc�A�1*

loss�	�;0�pL       �	����Yc�A�1*

loss�5�<��k�       �	�I��Yc�A�1*

losscS.;J�ӝ       �	����Yc�A�1*

lossk�<�YQ       �	D���Yc�A�1*

loss��W;���a       �	�2��Yc�A�1*

lossi�n<�\�u       �	����Yc�A�1*

loss�	<��{       �	�s��Yc�A�1*

lossҐt;��%       �	���Yc�A�1*

lossʕY=�Vr9       �	���Yc�A�1*

loss��<P�^       �	�m��Yc�A�1*

loss1=��9S       �	[$��Yc�A�1*

loss�`@;�z��       �	����Yc�A�1*

loss;�<��-�       �	�P��Yc�A�1*

loss��;3���       �	����Yc�A�1*

loss@��;is]       �	$���Yc�A�1*

lossZ�k=l�       �	*� �Yc�A�1*

loss!F�:���        �	�>�Yc�A�1*

loss�.W=�[_^       �	���Yc�A�1*

loss�� <Ȅ;h       �	�u�Yc�A�1*

loss(PO=2q��       �	3�Yc�A�1*

lossf�N<����       �	1��Yc�A�1*

loss�c<��8�       �	1x�Yc�A�1*

loss��<�?�       �	��Yc�A�1*

loss��I<7��Z       �	��Yc�A�1*

loss�9�<�       �	�>�Yc�A�1*

loss��<XTX       �	���Yc�A�1*

lossj��;TE�       �	��Yc�A�1*

loss3B�<q��       �	q��Yc�A�1*

loss�z=�Z)       �	�>	�Yc�A�1*

loss���<��$       �	��	�Yc�A�1*

loss�Es<���       �	C�Yc�A�1*

loss[:<% �6       �	x��Yc�A�1*

loss�p�<���~       �	.7�Yc�A�1*

loss��;�ZB`       �	4��Yc�A�1*

loss��><��)       �	�u�Yc�A�1*

loss=��=̱�4       �	��Yc�A�1*

loss.΃<�O�       �	i��Yc�A�1*

loss��=u�       �	B@�Yc�A�1*

loss��=��By       �	���Yc�A�1*

loss��<�=C       �	Xp�Yc�A�1*

loss���;;xR       �	��Yc�A�1*

loss�<2��       �	O��Yc�A�1*

lossd�$:�U�       �	Fx�Yc�A�1*

loss+<�.q       �	'�Yc�A�1*

loss�+X=g"��       �	@��Yc�A�1*

lossc�b=��#�       �	9��Yc�A�1*

lossd=�<�/�       �	Ϣ�Yc�A�1*

loss6ӄ<�t��       �	�C�Yc�A�1*

lossd�;=�a�       �	� �Yc�A�1*

loss4�9�3\       �	���Yc�A�1*

loss�u.:,r\J       �	�F�Yc�A�1*

loss�b=���       �	��Yc�A�1*

loss
�=�P�       �	���Yc�A�1*

loss�m�;��t�       �	�}�Yc�A�1*

loss�B�=��6�       �	(�Yc�A�1*

lossm=r��       �	,��Yc�A�1*

loss6XS</��       �	���Yc�A�1*

loss=��=�Z�3       �	+�Yc�A�1*

loss��H<x�%�       �	���Yc�A�1*

loss��;�O�)       �	]k�Yc�A�1*

loss���:e85V       �	(�Yc�A�1*

lossʾ�;�&e       �	���Yc�A�1*

loss�"=��       �	�` �Yc�A�1*

loss�п<c���       �	$!�Yc�A�1*

loss͖'=NO-�       �	�!�Yc�A�1*

loss���<8U�$       �	�W"�Yc�A�1*

lossj-<�R5�       �	� #�Yc�A�1*

loss�?4<��T�       �	j�#�Yc�A�1*

lossV�<6�pY       �	�F$�Yc�A�1*

losst�5<�-:�       �	��$�Yc�A�1*

loss��;B��       �	��%�Yc�A�1*

loss�=c<�MS�       �	\T&�Yc�A�1*

loss��<R&��       �	�&�Yc�A�1*

loss��8=�`�       �	��'�Yc�A�1*

loss�q=aT       �	�B(�Yc�A�1*

loss5�=K���       �	��(�Yc�A�1*

lossé�<���       �	)�)�Yc�A�1*

loss��;�d�       �	�C*�Yc�A�1*

loss�\�<�+(w       �	�*�Yc�A�1*

loss�K4=!.1       �	�+�Yc�A�1*

loss�5g<u� �       �	�,�Yc�A�1*

loss8.5=�O�       �	Z�,�Yc�A�1*

loss�;�;f��1       �	�N-�Yc�A�1*

loss/��;r��0       �	/�-�Yc�A�1*

loss�d�:���6       �	��.�Yc�A�1*

loss8�=8
d�       �	(/�Yc�A�1*

lossO�=?=w       �	�0�Yc�A�1*

loss���<�}Y=       �	�0�Yc�A�1*

lossJl�<%�A       �	�:1�Yc�A�1*

loss��=�zT       �	(�1�Yc�A�1*

loss<[�;f�`r       �	�t2�Yc�A�1*

loss��<y|&�       �	83�Yc�A�1*

loss��=�O3       �	��3�Yc�A�1*

lossZ��;]��4       �	@M4�Yc�A�1*

loss��;�(�P       �	��4�Yc�A�1*

loss�t<�7Y\       �	��5�Yc�A�1*

loss���;���K       �	�$6�Yc�A�1*

loss��<;�-X       �	v�6�Yc�A�1*

loss�jw<�VTR       �	=_7�Yc�A�1*

loss)c�<��h       �	�*8�Yc�A�1*

loss���;R{<       �	Z�8�Yc�A�1*

loss
X�:: ��       �	�w9�Yc�A�1*

loss-�"<�'K       �	:�Yc�A�1*

loss
-g<ϟ��       �	�:�Yc�A�1*

loss���;��#       �	�b;�Yc�A�1*

loss=�V�       �	�<�Yc�A�1*

loss��;S��       �	ʩ<�Yc�A�1*

lossD�;*�Z       �	�U=�Yc�A�1*

loss�t�;l�Ks       �	 �=�Yc�A�1*

loss(�;�5כ       �	I�>�Yc�A�2*

loss �n;��!	       �	q:?�Yc�A�2*

lossM�<�3�W       �	��?�Yc�A�2*

loss�.<��\       �	r@�Yc�A�2*

loss�O
<��!�       �	�A�Yc�A�2*

loss�$<���       �	��A�Yc�A�2*

loss��*<):U       �	�B�Yc�A�2*

lossT�3=�t)�       �	�2C�Yc�A�2*

loss��<��E       �	4�C�Yc�A�2*

loss��<0%@       �	��D�Yc�A�2*

loss��<�d|       �	D3E�Yc�A�2*

loss62=aЅ�       �	��E�Yc�A�2*

loss��=���       �	F�Yc�A�2*

loss���;(u�3       �	�'G�Yc�A�2*

loss�M0=&x�9       �	��G�Yc�A�2*

loss�r�<ə��       �	�lH�Yc�A�2*

loss}7�;���f       �	mI�Yc�A�2*

loss�:��_       �	�I�Yc�A�2*

loss��<b���       �	��K�Yc�A�2*

loss�~�<RӮ�       �	�L�Yc�A�2*

loss�&�<��xA       �	_M�Yc�A�2*

loss�F;>7V       �	�!N�Yc�A�2*

loss@��<k���       �	�O�Yc�A�2*

loss�v];e!(       �	��O�Yc�A�2*

loss���=��H       �	��P�Yc�A�2*

lossۏ�<����       �	8.Q�Yc�A�2*

lossvD�<�L��       �	��Q�Yc�A�2*

loss�-<誥       �	D�R�Yc�A�2*

lossYM�<��5B       �	�$S�Yc�A�2*

loss��e<��'       �	O�S�Yc�A�2*

lossSɺ<n��       �	&mT�Yc�A�2*

loss�Z�<J�^�       �	�U�Yc�A�2*

loss�r�<2Й�       �	%�U�Yc�A�2*

loss� <;��       �	HlV�Yc�A�2*

loss�O<�=d�       �	�
W�Yc�A�2*

loss��<ͩ�       �	D�W�Yc�A�2*

losss�K<`2a�       �	�GX�Yc�A�2*

lossjC�<���y       �	��X�Yc�A�2*

loss�Ό;b0+       �	��Y�Yc�A�2*

loss$��;%��W       �	y=Z�Yc�A�2*

loss?/�<��+�       �	��Z�Yc�A�2*

lossh=2m�       �	d�[�Yc�A�2*

lossx�=�
�       �	bM\�Yc�A�2*

lossX�<�g       �	5�\�Yc�A�2*

loss�ȑ<���e       �	��]�Yc�A�2*

loss|��:.BS       �	5&^�Yc�A�2*

loss�%=:�q�       �	i�^�Yc�A�2*

loss���=�;       �	8e_�Yc�A�2*

lossE�<�ȑ5       �	$`�Yc�A�2*

loss���<���:       �	x�`�Yc�A�2*

loss]��<��       �	5�a�Yc�A�2*

loss�t<��R�       �	yb�Yc�A�2*

lossJ�?:�`       �	c�Yc�A�2*

lossq<-m0�       �	ظc�Yc�A�2*

loss�<�K       �	�Ud�Yc�A�2*

loss�C=SD�       �	�e�Yc�A�2*

loss�c�<���       �	X�e�Yc�A�2*

loss!��<:в�       �	Gf�Yc�A�2*

loss�Զ<�>g       �	��f�Yc�A�2*

loss���<���Q       �	n�g�Yc�A�2*

lossC�_<�d��       �	�6h�Yc�A�2*

loss=H�<�3t�       �	��h�Yc�A�2*

loss�t�;U^�       �	9�i�Yc�A�2*

loss�E�;83��       �	�Bj�Yc�A�2*

loss&�>���       �	"�j�Yc�A�2*

losst��=�ѕ�       �	z�k�Yc�A�2*

loss59�<��       �	�Ll�Yc�A�2*

loss�gd=����       �	��l�Yc�A�2*

loss��<)"�l       �	r�m�Yc�A�2*

loss}X�;x�N�       �	>Yn�Yc�A�2*

loss��?;*���       �	�o�Yc�A�2*

loss�@r=2�l       �	:�o�Yc�A�2*

loss� 3;H>�G       �	��p�Yc�A�2*

loss��;�J)�       �	�4q�Yc�A�2*

loss� &=��eB       �	>�q�Yc�A�2*

loss��b;�:�X       �	;pr�Yc�A�2*

lossϢ�;���9       �	�s�Yc�A�2*

lossd�g:c|J�       �	P�s�Yc�A�2*

lossں�<���}       �	MLt�Yc�A�2*

loss3n�: �       �	W�t�Yc�A�2*

lossVcp=��       �	��u�Yc�A�2*

loss_�\=�T�%       �	 'v�Yc�A�2*

loss���<1v��       �	�v�Yc�A�2*

loss��;n5W       �	sgw�Yc�A�2*

loss��;9���       �	�x�Yc�A�2*

loss�=�*�C       �	{�x�Yc�A�2*

loss��@=�r��       �	�y�Yc�A�2*

lossa]K=�#'�       �	)z�Yc�A�2*

loss�|�=���p       �	/�z�Yc�A�2*

loss㳍;ƒ�"       �	GY{�Yc�A�2*

loss��<����       �	R�{�Yc�A�2*

loss��<2��
       �	�|�Yc�A�2*

loss���<�b��       �	/Q}�Yc�A�2*

loss���;@���       �	��}�Yc�A�2*

lossԴ�:�-F�       �	#�Yc�A�2*

loss��;2c@n       �	�g��Yc�A�2*

loss��f;�Q�0       �	���Yc�A�2*

loss~`�< �x       �	}���Yc�A�2*

loss��;3�I�       �	,J��Yc�A�2*

loss4�<}rt       �	 䂌Yc�A�2*

lossNs�<�o_        �	f���Yc�A�2*

lossz�=<OGip       �	�=��Yc�A�2*

loss
��9�e@�       �	|ф�Yc�A�2*

loss�[P;�%^�       �	�u��Yc�A�2*

loss�6=*�2�       �	���Yc�A�2*

loss�y�;3_:       �	Ի��Yc�A�2*

loss�!<N�*       �	�a��Yc�A�2*

loss�	<1�#�       �	>��Yc�A�2*

loss�`�;�	p_       �	ߦ��Yc�A�2*

loss5܊<�#�       �	�@��Yc�A�2*

loss�<M�C�       �	�؉�Yc�A�2*

loss�5;�˝\       �	�x��Yc�A�2*

lossճ<��N       �	7��Yc�A�2*

loss��V=�N͋       �	S���Yc�A�2*

loss�϶<ou =       �	�W��Yc�A�2*

loss��<;�j       �	����Yc�A�2*

lossI�<���       �	Օ��Yc�A�2*

lossn� =��       �	(H��Yc�A�2*

loss�� ;V��|       �	*ᎌYc�A�2*

lossz�O; �N       �	>{��Yc�A�2*

loss�ٜ<wm�h       �	�3��Yc�A�2*

loss��;%���       �	�*��Yc�A�2*

loss�V?<m��M       �	q⑌Yc�A�2*

loss���:	��       �	΍��Yc�A�2*

loss�#�</�#       �	er��Yc�A�2*

lossΎ=�!V2       �	�2��Yc�A�2*

loss8�=j8G       �	�䔌Yc�A�2*

loss�i�<�.��       �	�ו�Yc�A�2*

losshc	;��<�       �	�p��Yc�A�3*

loss[lj=����       �	��Yc�A�3*

loss�[;X��       �	众�Yc�A�3*

loss�K;�ЁX       �	@l��Yc�A�3*

lossѽ=T.       �	��Yc�A�3*

loss��;)y�)       �	O���Yc�A�3*

loss)�:2�7�       �	�P��Yc�A�3*

loss��;���       �	��Yc�A�3*

lossW��;�sK`       �	I���Yc�A�3*

loss�:�I       �	AG��Yc�A�3*

loss֊;�gJ@       �	�圌Yc�A�3*

lossj�	;�*Ow       �	����Yc�A�3*

loss ��<��*x       �	�8��Yc�A�3*

loss�9�:�       �	�㞌Yc�A�3*

loss�p�8�j�)       �	y[��Yc�A�3*

loss�K:�y>D       �	Q���Yc�A�3*

loss��;щ�!       �	����Yc�A�3*

loss��<��       �	�5��Yc�A�3*

loss��;.cc�       �	�Ӣ�Yc�A�3*

lossRݥ8�A       �	R}��Yc�A�3*

losss�(:	�|       �	�i��Yc�A�3*

loss���=8|�:       �	���Yc�A�3*

lossT�&;�("       �	���Yc�A�3*

lossl�>z�       �	YL��Yc�A�3*

loss�4=��x:       �	�榌Yc�A�3*

loss	@<Ԗ%"       �	����Yc�A�3*

loss6��:���       �	�*��Yc�A�3*

loss�(�<�@g(       �	)˨�Yc�A�3*

loss���;>���       �	k��Yc�A�3*

loss��C=κ�       �	�
��Yc�A�3*

lossË�;Gi�       �	/���Yc�A�3*

loss���<�x��       �	W��Yc�A�3*

lossn�:ε�n       �	����Yc�A�3*

losshZp<A�ja       �	����Yc�A�3*

loss���<J4	x       �	/���Yc�A�3*

loss�sx;��a�       �	g���Yc�A�3*

loss�3�<�s0�       �	 ���Yc�A�3*

loss�^(=n��       �	u��Yc�A�3*

lossMn6=���       �	�-��Yc�A�3*

lossD�H<��p       �	9���Yc�A�3*

lossv�#<�}g"       �	ظ��Yc�A�3*

lossqL<`��v       �	�ǵ�Yc�A�3*

lossM�<~f8�       �	�Yc�A�3*

loss�m�<���X       �	�׷�Yc�A�3*

loss]I<=.	       �	�Ѹ�Yc�A�3*

lossG��:��.�       �	�3��Yc�A�3*

loss�L�;��)�       �	hX��Yc�A�3*

loss젌;5�X       �	ϻ��Yc�A�3*

loss=�<��*       �	 ��Yc�A�3*

loss�.S;�uc4       �	�9��Yc�A�3*

loss#��<ǈb.       �	<ܿ�Yc�A�3*

loss�1q=����       �	ӈ��Yc�A�3*

lossH�I<b��       �	.��Yc�A�3*

loss��F<W�r=       �	����Yc�A�3*

lossXX�;9}�R       �	x�Yc�A�3*

loss���9�]B       �	�@ÌYc�A�3*

lossiب<��K       �	�ÌYc�A�3*

lossJ��;��~4       �	��ČYc�A�3*

loss)�-=z�S�       �	�ŌYc�A�3*

loss��<ft�       �	"�ŌYc�A�3*

loss\�h=�<~�       �	�^ƌYc�A�3*

loss2��<L2c�       �	FǌYc�A�3*

lossx
<`���       �	�ǌYc�A�3*

lossw��<r���       �	�dȌYc�A�3*

loss>m<��"�       �	��ȌYc�A�3*

loss�D;�W�       �	��ɌYc�A�3*

lossʟ=A�р       �	sʌYc�A�3*

loss��=��B?       �	CˌYc�A�3*

lossA(=��        �	j�ˌYc�A�3*

loss�{�;o���       �	RěYc�A�3*

loss���<��:       �	͌Yc�A�3*

loss��g<�X}       �	��͌Yc�A�3*

loss��;=�;B       �	-`ΌYc�A�3*

loss�* =�c��       �	y�	�Yc�A�3*

loss<�L<�(5       �	�R�Yc�A�3*

loss+B<�;�       �	���Yc�A�3*

loss��E<p�n�       �	�X�Yc�A�3*

loss�Q�<���       �	n��Yc�A�3*

lossq	f;�U~�       �	_F�Yc�A�3*

loss9T<�b!       �	���Yc�A�3*

loss��|<�.wj       �	���Yc�A�3*

loss�]�=x�E        �	\��Yc�A�3*

loss J�;�<�k       �	wf�Yc�A�3*

loss�;�͠y       �	���Yc�A�3*

loss�wB;n��8       �	��Yc�A�3*

loss!�:=<G��       �	n�Yc�A�3*

lossV��;���K       �	W��Yc�A�3*

loss$'4<�A       �	~��Yc�A�3*

loss���<�0��       �	`[�Yc�A�3*

loss��9�>]@       �	D��Yc�A�3*

lossJ�<1v~g       �	�1!�Yc�A�3*

lossi�;��       �	2"�Yc�A�3*

losslj	<�@�{       �	ys#�Yc�A�3*

loss2�'=�Z�       �	�%�Yc�A�3*

loss	>�=gΩ�       �	�9&�Yc�A�3*

loss���:�C�M       �	�Z'�Yc�A�3*

loss�U\=W}       �	��(�Yc�A�3*

lossS��:����       �	o*�Yc�A�3*

loss��F<N�?u       �	�+�Yc�A�3*

loss|S�<R+q�       �	�c,�Yc�A�3*

loss�T*;�ۭ^       �	�-�Yc�A�3*

lossa[�;�m�       �	K!/�Yc�A�3*

loss�!!<x0rb       �	y0�Yc�A�3*

loss7�G<����       �	61�Yc�A�3*

loss
~=�q       �	LR2�Yc�A�3*

loss���<��x�       �	4�3�Yc�A�3*

lossL=p�       �	L�4�Yc�A�3*

loss�uF;�sn�       �	a76�Yc�A�3*

loss�y�;n
�       �	�7�Yc�A�3*

loss�q;��L       �	39�Yc�A�3*

loss1n<��BT       �	�F:�Yc�A�3*

loss�=g*;       �	�{;�Yc�A�3*

lossT��;�OF�       �	��<�Yc�A�3*

loss��;>U       �	�>�Yc�A�3*

loss��<e��o       �	II?�Yc�A�3*

lossN��:���H       �	8f@�Yc�A�3*

loss�&<e��Y       �	h�A�Yc�A�3*

loss��5;�f̝       �	��B�Yc�A�3*

loss��<��D�       �	pD�Yc�A�3*

loss��=��ʝ       �	��E�Yc�A�3*

lossj�<<��e       �	,EG�Yc�A�3*

loss ��<]CC�       �	=aH�Yc�A�3*

losse�<��w       �	b�I�Yc�A�3*

loss��;��k�       �	�K�Yc�A�3*

loss;�<���-       �	amL�Yc�A�3*

lossD
�;�je�       �	K�M�Yc�A�3*

lossA��=��       �	`O�Yc�A�3*

loss
<�;^s�       �	L�P�Yc�A�3*

loss%y�:FU��       �	��Q�Yc�A�4*

loss�w;��9�       �	�*S�Yc�A�4*

lossX7�:��v%       �	�T�Yc�A�4*

loss�V<%|u�       �	6�U�Yc�A�4*

loss)-�<9al%       �	V-W�Yc�A�4*

loss�Bs=��U�       �	_�X�Yc�A�4*

loss���;s)��       �	6Z�Yc�A�4*

loss�g�:�~��       �	^f[�Yc�A�4*

loss�n�<�]q       �	h�\�Yc�A�4*

loss�B�;3x�       �	��_�Yc�A�4*

loss��;l+�       �	4a�Yc�A�4*

loss��=t��       �	�6b�Yc�A�4*

loss)�<p���       �	��c�Yc�A�4*

loss��W<!@va       �	e�Yc�A�4*

loss��;	>�       �	Sf�Yc�A�4*

loss�O<6��       �	6�g�Yc�A�4*

loss��;�9e       �	�i�Yc�A�4*

loss-�4<�n�       �	�Jj�Yc�A�4*

loss�C<~��y       �	E�k�Yc�A�4*

loss�0=��I       �	�9m�Yc�A�4*

lossL=<=x	�       �	Ŏn�Yc�A�4*

lossh+< �_       �	k�o�Yc�A�4*

loss,��<�`       �	]q�Yc�A�4*

loss&#B=x���       �	dxr�Yc�A�4*

loss��v<2���       �	��s�Yc�A�4*

loss��~<l��       �	��t�Yc�A�4*

loss#��<�D�y       �	�[v�Yc�A�4*

loss͎�;�E��       �	��w�Yc�A�4*

loss�=�::<�y       �	��x�Yc�A�4*

loss��\<��       �	�/z�Yc�A�4*

lossW�R=
�=�       �	O�{�Yc�A�4*

loss2�w<Z9�       �	��|�Yc�A�4*

loss<? <ď�       �	�)~�Yc�A�4*

loss�o�<r��       �	��Yc�A�4*

loss��2=N[�u       �	�倍Yc�A�4*

loss��<��j       �	�=��Yc�A�4*

lossH��<�;�       �	�v��Yc�A�4*

loss=O8=��       �	����Yc�A�4*

loss��
;���       �	7���Yc�A�4*

lossz�G<�-IS       �	m8��Yc�A�4*

loss,�;��>Z       �	6���Yc�A�4*

loss�\�<%1�y       �	�D��Yc�A�4*

loss�J<v1h       �	Fy��Yc�A�4*

loss�=���       �	m���Yc�A�4*

loss��<	B��       �	U�Yc�A�4*

loss5�;[�M;       �	����Yc�A�4*

loss&�;���       �	�4��Yc�A�4*

loss%$�=�=�       �	����Yc�A�4*

loss؈:��       �	�Ó�Yc�A�4*

lossH�$<r
.�       �	O͔�Yc�A�4*

loss��q=�ar       �	�蕍Yc�A�4*

loss�$@:BY       �	�B��Yc�A�4*

loss�@U=��T       �	̗��Yc�A�4*

lossC�<��Iu       �	ͮ��Yc�A�4*

loss�
<\	�N       �	��Yc�A�4*

loss��@= '�       �	XU��Yc�A�4*

loss&�	;F�4       �	X���Yc�A�4*

loss�a�=��L�       �	TȞ�Yc�A�4*

loss��<>�]0       �	[ퟍYc�A�4*

loss�9`<GE��       �	�V��Yc�A�4*

lossa��<���~       �	D���Yc�A�4*

loss��,=v{%       �	]���Yc�A�4*

lossq�=5��`       �	���Yc�A�4*

loss;�=�~��       �	����Yc�A�4*

loss>�=��K       �	���Yc�A�4*

loss�us;�PZ<       �	�&��Yc�A�4*

loss�=>��       �	z���Yc�A�4*

loss � <�~�       �	L��Yc�A�4*

lossi<���       �	~8��Yc�A�4*

loss7�;�-P       �	����Yc�A�4*

loss��=7�*�       �	a���Yc�A�4*

loss2�;�        �	!?��Yc�A�4*

loss�
T<��%       �	����Yc�A�4*

loss�8�<௃�       �	W곍Yc�A�4*

lossz5�<[��V       �	LR��Yc�A�4*

loss�:�;Iiy@       �	�C��Yc�A�4*

lossS9�<,��       �	����Yc�A�4*

loss㼚<��¡       �	���Yc�A�4*

loss_�};��H!       �	����Yc�A�4*

loss8��<�_-       �	����Yc�A�4*

lossv<>��       �	\㼍Yc�A�4*

loss��<���       �	�N��Yc�A�4*

loss|��<���       �	���Yc�A�4*

loss�=�<Jj�_       �	-��Yc�A�4*

loss��:ֺ�S       �	�hYc�A�4*

loss�*�:D���       �	=�ÍYc�A�4*

loss��;�x�       �	��čYc�A�4*

loss��<af�c       �	�3ƍYc�A�4*

lossq��<�ּ~       �	��ǍYc�A�4*

lossaZ+=B-�       �	�4ɍYc�A�4*

loss��<!;mP       �	�ʍYc�A�4*

lossz�=�;       �	�ZˍYc�A�4*

loss���;�'��       �	A�̍Yc�A�4*

loss���;v�4�       �	� ΍Yc�A�4*

loss��)<6��       �	h�΍Yc�A�4*

loss ��;�d       �	?;ЍYc�A�4*

loss��<���       �	��эYc�A�4*

loss��|=G��$       �	T9ӍYc�A�4*

loss��;���       �	�/ԍYc�A�4*

loss/e�<�ڵo       �	IՍYc�A�4*

lossNɕ9�:�y       �	\�֍Yc�A�4*

lossOԇ;�X�       �	9؍Yc�A�4*

loss)tG<
z�w       �	�-ٍYc�A�4*

loss�a~=o�/2       �	�qڍYc�A�4*

loss���:���       �	��ۍYc�A�4*

lossĖ�<�v�       �	(ݍYc�A�4*

loss��<.���       �	7pލYc�A�4*

loss)�=�8(       �	��ߍYc�A�4*

loss�8�:����       �	<K�Yc�A�4*

loss��L<yPWT       �	�5�Yc�A�4*

loss��);նU�       �	g`�Yc�A�4*

loss�K;#�*       �	I��Yc�A�4*

loss���<����       �	m�Yc�A�4*

lossv	<��Q&       �	V+�Yc�A�4*

loss-@1<f�R�       �	�g�Yc�A�4*

lossd��=9C�f       �	���Yc�A�4*

loss��Q;l# �       �	��Yc�A�4*

loss���:1�D�       �	��Yc�A�4*

loss��7={��E       �	�^�Yc�A�4*

loss�d;s�۠       �	��Yc�A�4*

loss�h2<F��       �	$��Yc�A�4*

loss�+�<&\ O       �	k�Yc�A�4*

loss�V�<_'�~       �	W�Yc�A�4*

lossjї:��6�       �	N��Yc�A�4*

loss_�;Ĵ<s       �	��Yc�A�4*

loss��;����       �	�R��Yc�A�4*

loss\Ľ=Θ�-       �	y��Yc�A�4*

loss1��<W�        �	J��Yc�A�4*

lossJ<;��T�       �	R���Yc�A�5*

lossn��;`�@R       �	����Yc�A�5*

loss�?p;@۾�       �	�K��Yc�A�5*

loss�p�<L �       �	ڮ��Yc�A�5*

loss
��<V	;       �	�� �Yc�A�5*

loss/�;��       �	nn�Yc�A�5*

loss=�V= �       �	���Yc�A�5*

loss�O�<�F�       �	;��Yc�A�5*

loss@�<�� o       �	U�Yc�A�5*

loss}ֲ;!%�       �	8��Yc�A�5*

lossJ#5<���D       �	O	�Yc�A�5*

lossh�<n��K       �	�`
�Yc�A�5*

loss�T�<�F�       �	��Yc�A�5*

lossoCI;oy��       �	
��Yc�A�5*

loss)7�:��b       �	��Yc�A�5*

lossVR�<�,ю       �	�
�Yc�A�5*

loss��F<-��       �	g��Yc�A�5*

loss�͍<0�k�       �	��Yc�A�5*

loss>X<, ;�       �	
g�Yc�A�5*

loss�H�<�M��       �	�Yc�A�5*

loss���<e�}       �	�]�Yc�A�5*

lossZ��;T�S�       �	�}�Yc�A�5*

loss�]=���(       �	��Yc�A�5*

loss�;����       �	>Z�Yc�A�5*

loss��+<Q�jA       �	|�Yc�A�5*

loss �$<���       �	���Yc�A�5*

loss%�#<d��       �	_��Yc�A�5*

loss��s=�Sv�       �	�� �Yc�A�5*

loss�k�;x	�       �	K"�Yc�A�5*

loss$&;!��       �	�=#�Yc�A�5*

loss:��;gK�       �	Q�$�Yc�A�5*

loss�V@<>]5       �	�%�Yc�A�5*

loss��<��Ǐ       �	�S'�Yc�A�5*

loss�X�=���N       �	��(�Yc�A�5*

loss�H�<����       �	�U*�Yc�A�5*

loss���<���m       �		�+�Yc�A�5*

loss�Y<���       �	�;-�Yc�A�5*

loss�q<���!       �	��.�Yc�A�5*

loss�];=��C       �	��/�Yc�A�5*

lossm8'< �j       �	�?1�Yc�A�5*

loss�HS=�Gv"       �	}�2�Yc�A�5*

loss���;�M�       �	I�3�Yc�A�5*

loss~�
<}{�       �	�Y5�Yc�A�5*

loss��;�_e{       �	^�6�Yc�A�5*

loss��N;s���       �	�\8�Yc�A�5*

loss�d&<7�P�       �	��9�Yc�A�5*

loss>�:�Vs$       �	��:�Yc�A�5*

loss�Qe;�?N�       �	�[<�Yc�A�5*

loss�E�9lP�       �	-y=�Yc�A�5*

loss�k�<d6       �	��>�Yc�A�5*

loss6_.;�쥰       �	�/@�Yc�A�5*

loss�w=�3��       �	��A�Yc�A�5*

lossյ:yJ,a       �	��B�Yc�A�5*

loss��<"��       �	�:D�Yc�A�5*

loss�Ɛ<+�       �	C�E�Yc�A�5*

loss��*<Y�cf       �	�G�Yc�A�5*

loss�V�:���       �	pH�Yc�A�5*

loss,��;���F       �	��I�Yc�A�5*

lossϖZ=qی       �	�oK�Yc�A�5*

loss|9�;�Z��       �	�wL�Yc�A�5*

loss8́<�F��       �	}�M�Yc�A�5*

lossJb�=�=Hu       �	!YO�Yc�A�5*

loss�=�#�       �	��P�Yc�A�5*

lossI�:�&��       �	�Q�Yc�A�5*

loss��9=�       �	��R�Yc�A�5*

loss��<����       �	�`T�Yc�A�5*

loss.q�;
��       �	�U�Yc�A�5*

lossi�B=���       �	Z�V�Yc�A�5*

loss�<ƵĂ       �	�UX�Yc�A�5*

loss���:�_�       �	'�Y�Yc�A�5*

loss��7:��       �	r�Z�Yc�A�5*

loss���:_��       �	<�[�Yc�A�5*

loss�8�;� �       �	V]�Yc�A�5*

lossR`l<�a       �	�^�Yc�A�5*

lossl�<�(�       �	<�_�Yc�A�5*

loss�;��-�       �	
ga�Yc�A�5*

loss��;U�3       �	!�b�Yc�A�5*

lossN��;�A��       �	E�c�Yc�A�5*

loss �<�SA�       �	�1e�Yc�A�5*

loss\:�<���q       �	z�f�Yc�A�5*

loss�L�;8.��       �	��g�Yc�A�5*

loss"j;��3"       �	.i�Yc�A�5*

loss�`�;E��,       �	�'j�Yc�A�5*

loss���;#��	       �	��k�Yc�A�5*

lossh|�=�G�       �	��l�Yc�A�5*

lossQ`�;���{       �	n�Yc�A�5*

loss�= ܝ�       �	OYo�Yc�A�5*

loss��;���       �	@�p�Yc�A�5*

loss�;�;ha��       �	��q�Yc�A�5*

loss4�V=�6JY       �	Js�Yc�A�5*

loss ��;���       �	Ot�Yc�A�5*

loss��=�RM       �	��u�Yc�A�5*

loss��<b"�       �	�w�Yc�A�5*

lossA��;xʊA       �	&x�Yc�A�5*

lossT_<���f       �	cdy�Yc�A�5*

loss7�;)ϙ       �	+�z�Yc�A�5*

loss�W=Q'�'       �	�@|�Yc�A�5*

lossԝ�9�%I#       �	b}�Yc�A�5*

loss��;�G       �	��~�Yc�A�5*

loss)|�:C��B       �	S ��Yc�A�5*

loss%WQ=���       �	@3��Yc�A�5*

loss�;=�tA       �	�p��Yc�A�5*

loss}�J<�Ն�       �	�փ�Yc�A�5*

loss̛�<�K��       �	o��Yc�A�5*

lossLV
;lٺ9       �	�;��Yc�A�5*

loss{��;7�h�       �	&S��Yc�A�5*

loss�l�<�`�H       �	@܈�Yc�A�5*

loss���:�9�y       �	��Yc�A�5*

lossv��;۫��       �	�L��Yc�A�5*

loss�1�:��       �	H���Yc�A�5*

loss-Z�:G�$�       �	ƍ�Yc�A�5*

loss/�<�'Ɋ       �	�쎎Yc�A�5*

lossa�<=�N9�       �	�D��Yc�A�5*

loss�<=.$P       �	H���Yc�A�5*

loss���<^��        �	�W��Yc�A�5*

lossJ��<�:��       �	�n��Yc�A�5*

loss}��<�ǫ       �	����Yc�A�5*

loss��;<�DW       �	��Yc�A�5*

loss�z;[ڈ       �	�.��Yc�A�5*

loss��)=��j       �	:��Yc�A�5*

loss��=�4�0       �	�x��Yc�A�5*

loss�o�;?Q��       �	7ݛ�Yc�A�5*

loss�]�<�~�y       �	��Yc�A�5*

loss�`&:��R�       �	qU��Yc�A�5*

loss�@s<��@I       �	L���Yc�A�5*

loss�q
<�:��       �	k��Yc�A�5*

loss6�<P%�       �	��Yc�A�5*

loss1;P$��       �	�c��Yc�A�5*

loss*�;\ۙ       �	����Yc�A�6*

lossPM=s���       �	$&��Yc�A�6*

loss؋�;�Gh       �	G;��Yc�A�6*

loss�i;X�~�       �	.���Yc�A�6*

loss��=���       �	���Yc�A�6*

loss��<z�2�       �	|G��Yc�A�6*

loss}Q;��R       �	G���Yc�A�6*

loss}�<�	�       �	T���Yc�A�6*

lossU<���       �	�d��Yc�A�6*

loss��?;���       �	�)��Yc�A�6*

loss���;EO       �	հ�Yc�A�6*

loss��<7�B�       �	�~��Yc�A�6*

loss�0e;M�C�       �	�)��Yc�A�6*

loss��2<��+m       �	β�Yc�A�6*

loss��;��?       �	�z��Yc�A�6*

loss���;�{       �	��Yc�A�6*

loss�QO=K� n       �	�Ĵ�Yc�A�6*

losswfB<r!$b       �	�e��Yc�A�6*

losseAv<>(d�       �	��Yc�A�6*

loss�j;q !       �	ޯ��Yc�A�6*

loss�;3'�;       �	�p��Yc�A�6*

loss��j;����       �	���Yc�A�6*

loss�O�<�r-�       �	�ø�Yc�A�6*

loss;�t<�'h       �	�^��Yc�A�6*

loss_��:���S       �	����Yc�A�6*

lossA�<XRj       �	�r��Yc�A�6*

loss��;���g       �	b��Yc�A�6*

loss	�=�V��       �	����Yc�A�6*

lossL`�:���       �	|���Yc�A�6*

loss�$�;Sj!!       �	����Yc�A�6*

loss �|;���r       �	q˿�Yc�A�6*

loss�{;�6\h       �	TYc�A�6*

loss��0=2�j�       �	��Yc�A�6*

loss���<'~4       �	�gÎYc�A�6*

loss�;<�H�'       �	<ĎYc�A�6*

loss�N<�d�       �	:�ĎYc�A�6*

loss�ҹ<�S�       �	�^ŎYc�A�6*

loss�=v.��       �	e�ƎYc�A�6*

lossھ�<�;       �	!#ǎYc�A�6*

loss��:�J�       �	�ǎYc�A�6*

lossW{T;b	       �	_aȎYc�A�6*

loss�m;�v�       �		�ȎYc�A�6*

loss�u<cr�G       �	�ɎYc�A�6*

loss���;Yιa       �	�+ʎYc�A�6*

loss�.,<=o       �	��ʎYc�A�6*

loss�|;����       �	�XˎYc�A�6*

loss+ ;�H�F       �	��ˎYc�A�6*

loss@��;�>��       �	H�̎Yc�A�6*

lossl)�;ߧ��       �	�$͎Yc�A�6*

loss\M�;�I�       �	��͎Yc�A�6*

loss #<��^�       �	�hΎYc�A�6*

loss8��<�;Y�       �	��ΎYc�A�6*

loss�2�=�5�       �	�ώYc�A�6*

loss�h(;$��       �	E�ЎYc�A�6*

loss-+�=a�k�       �	"4юYc�A�6*

lossԲ<�]b       �	��юYc�A�6*

lossQ,�<��;       �	=_ҎYc�A�6*

loss��:�p�       �	�%ӎYc�A�6*

lossj �;���       �	½ӎYc�A�6*

loss��%=8,��       �	�ԎYc�A�6*

lossң�:��Q       �	eՎYc�A�6*

loss�\�;|9�g       �	�֎Yc�A�6*

loss��:B*5�       �	)\׎Yc�A�6*

loss_�;�F`�       �	�؎Yc�A�6*

loss��A=�\�       �	n�؎Yc�A�6*

lossU�<A�K       �	�uَYc�A�6*

lossC��<�P�       �	,�ڎYc�A�6*

lossRtL;�髁       �	�oێYc�A�6*

lossHK<���       �	�܎Yc�A�6*

loss� �<t���       �	~�܎Yc�A�6*

loss}P�:q�9�       �	mݎYc�A�6*

loss\�>�C�       �	ގYc�A�6*

lossZe�<��.       �	_
ߎYc�A�6*

loss	�f;��K       �	ۤߎYc�A�6*

loss�v�=3w�       �	
f��Yc�A�6*

loss�u
:�|��       �	�Yc�A�6*

loss ��<?�	-       �	���Yc�A�6*

losse��;�d@�       �	k�Yc�A�6*

loss�U0;~���       �	G�Yc�A�6*

lossQ�;��A6       �	N��Yc�A�6*

lossA��<� �       �	�'�Yc�A�6*

loss鎳<Z%T�       �	��Yc�A�6*

loss�W<#�0D       �	�S�Yc�A�6*

lossw�:�y�a       �	h��Yc�A�6*

loss�=�=qG�       �	o��Yc�A�6*

loss:(9��j       �	� �Yc�A�6*

loss73;�Cx       �	f��Yc�A�6*

loss�<S�s       �	rO�Yc�A�6*

lossA��<�0_       �	T��Yc�A�6*

loss�F};���p       �	���Yc�A�6*

loss���;���%       �	�F�Yc�A�6*

loss7�<ý<       �	y��Yc�A�6*

loss/��<Ѭ�8       �	�{�Yc�A�6*

loss���<@ܴ�       �	��Yc�A�6*

lossU\!<��P       �	a��Yc�A�6*

lossm�<�/Qh       �	U�Yc�A�6*

loss���;q�˸       �	5��Yc�A�6*

lossn�<]�z�       �	z��Yc�A�6*

lossL:D=��v"       �	���Yc�A�6*

loss�1�;�m�5       �	���Yc�A�6*

loss��+=z�h�       �	�j�Yc�A�6*

loss�$=C
�       �	.�Yc�A�6*

lossG5�<��v\       �	i��Yc�A�6*

lossW��;]�yY       �	 B�Yc�A�6*

lossO�W<�4�9       �	���Yc�A�6*

loss�h�:��       �	�s�Yc�A�6*

lossd �:��R       �	N��Yc�A�6*

loss���;�'�       �	���Yc�A�6*

loss_Y�<#��p       �	�C��Yc�A�6*

lossx�;���^       �	���Yc�A�6*

losswK�<�y�&       �	@���Yc�A�6*

loss/N=/ݙ�       �	BA��Yc�A�6*

lossȚ{<!�       �	#���Yc�A�6*

loss7+r<S�R�       �	�o��Yc�A�6*

loss�Z�<+�       �	�	��Yc�A�6*

losss3T<�N��       �	����Yc�A�6*

lossHA=�y��       �	�4��Yc�A�6*

losse��:֬ѐ       �	����Yc�A�6*

loss1�%=�d]a       �	/k��Yc�A�6*

loss؅<�
p!       �	�-��Yc�A�6*

loss1��<�ZL       �	Z���Yc�A�6*

loss�A�<��ӭ       �	i��Yc�A�6*

loss�d;��:       �	����Yc�A�6*

loss��$=2)u�       �	����Yc�A�6*

loss��R=(�!       �	� �Yc�A�6*

loss�W�;�\��       �	��Yc�A�6*

loss�u�<���       �	�L�Yc�A�6*

loss��t<D��       �	��Yc�A�6*

loss�w<i�       �	_��Yc�A�7*

loss;��;��X�       �	
+�Yc�A�7*

loss\�D=h���       �	���Yc�A�7*

loss� =l�s�       �	�S�Yc�A�7*

loss.�[<iR�       �	���Yc�A�7*

loss���:���       �	�w�Yc�A�7*

loss�п<>��Q       �	�(�Yc�A�7*

lossf�=e!]�       �	c��Yc�A�7*

lossI=,�R       �	q�Yc�A�7*

loss!J<��R�       �	�	�Yc�A�7*

lossR��;��1,       �	�a
�Yc�A�7*

lossq2'=��       �	��
�Yc�A�7*

loss�oN<�<�{       �	���Yc�A�7*

loss$~;vI]�       �	�&�Yc�A�7*

loss,c�;^�8       �	#��Yc�A�7*

loss��A=��+�       �	�R�Yc�A�7*

loss��%=�z�       �	���Yc�A�7*

loss���;�^<       �	~�Yc�A�7*

loss���<$�3       �	U�Yc�A�7*

lossj_K;��?W       �	X��Yc�A�7*

loss#9�;*̞       �	9C�Yc�A�7*

loss�'�;тR�       �	��Yc�A�7*

loss�C�9�'�       �	�u�Yc�A�7*

loss�c�;�ї/       �	[�Yc�A�7*

loss�_\=M��d       �	5��Yc�A�7*

loss��y;��K�       �	���Yc�A�7*

loss�:=�j       �	�8�Yc�A�7*

loss̇�;��}       �	S��Yc�A�7*

loss��;o!ڡ       �	�z�Yc�A�7*

loss�0<2��c       �	2;�Yc�A�7*

loss��;�>�       �	�v�Yc�A�7*

loss8�>W�n�       �	��Yc�A�7*

loss�=+<�Q�.       �	7��Yc�A�7*

loss1�<�
~*       �	&9�Yc�A�7*

loss��	<��X       �	l��Yc�A�7*

loss_�;��U/       �	�m�Yc�A�7*

loss	�z;�p��       �	e�Yc�A�7*

lossq�U=!>^)       �	���Yc�A�7*

loss���9(~�       �	�/�Yc�A�7*

loss4�;�긪       �	+��Yc�A�7*

loss���<��       �	X�Yc�A�7*

loss�>�<t �L       �	)&�Yc�A�7*

loss�8>^ax�       �	��Yc�A�7*

lossMȚ;�2BL       �	�R�Yc�A�7*

loss�4<[9��       �	���Yc�A�7*

loss��;U��       �	�{ �Yc�A�7*

loss�!3;�RY       �	8!�Yc�A�7*

lossѕ�:��       �	a�!�Yc�A�7*

lossq=�A�       �	%="�Yc�A�7*

losss�;�|��       �	��"�Yc�A�7*

loss;��<�)       �	e#�Yc�A�7*

loss�y<QS�u       �	��#�Yc�A�7*

loss�7.=-��i       �	\�$�Yc�A�7*

loss&J;'/*       �	4%�Yc�A�7*

loss�:	<�"��       �	��%�Yc�A�7*

lossF4:~���       �	�[&�Yc�A�7*

loss(��;���[       �	��&�Yc�A�7*

loss��{<6��       �	є'�Yc�A�7*

loss��<T"��       �	��(�Yc�A�7*

loss=��;���B       �	T)�Yc�A�7*

loss��<%t�       �	��)�Yc�A�7*

loss��;���w       �	YR*�Yc�A�7*

loss�Ƌ;#���       �	g�*�Yc�A�7*

loss��P;x��       �	��+�Yc�A�7*

loss��;��       �	29,�Yc�A�7*

loss���<���       �	�,�Yc�A�7*

loss��=��I       �	Pn-�Yc�A�7*

lossL�d;�Pe�       �	�.�Yc�A�7*

lossW�!:-�;*       �	��.�Yc�A�7*

loss���;����       �	�;/�Yc�A�7*

loss���<,vq       �	0�/�Yc�A�7*

lossa�:���       �	�|0�Yc�A�7*

loss==��)�       �	]1�Yc�A�7*

lossS�-:;٨)       �	ݴ1�Yc�A�7*

lossƾB;]�Z       �	�a2�Yc�A�7*

loss�*E<�S��       �	�3�Yc�A�7*

lossv�(;4-��       �	�3�Yc�A�7*

loss\�7<���R       �	�D4�Yc�A�7*

lossT�;~���       �	��4�Yc�A�7*

lossD�<>Ge�       �	�{5�Yc�A�7*

lossXJO<���       �	�6�Yc�A�7*

loss�F>:Z���       �	ҧ6�Yc�A�7*

loss�V;�nx       �	�<7�Yc�A�7*

loss��T;�Ú       �	x�7�Yc�A�7*

loss;��]�       �	s8�Yc�A�7*

loss}�;)s��       �	9�Yc�A�7*

loss�;U|<       �	@�9�Yc�A�7*

loss�o;����       �	H7:�Yc�A�7*

loss}+�<����       �	!�:�Yc�A�7*

loss	=��`�       �	�k;�Yc�A�7*

loss��Q<7͔       �	�<�Yc�A�7*

loss�
B<烾       �	�<�Yc�A�7*

loss:�=d��7       �	R,=�Yc�A�7*

loss���:�]�       �	Ӿ=�Yc�A�7*

loss�8ia�5       �	S>�Yc�A�7*

lossq�<��$�       �	i�>�Yc�A�7*

loss��6; �h       �	܃?�Yc�A�7*

loss��;by�       �	ς@�Yc�A�7*

loss�ES=NŖ       �	�A�Yc�A�7*

lossQX�9���F       �	&�A�Yc�A�7*

loss[��<��}       �	�IB�Yc�A�7*

loss��9��H       �	��B�Yc�A�7*

losseo�7�g�Q       �	�rC�Yc�A�7*

loss�U;;��^�       �	�D�Yc�A�7*

lossA�;�;O*       �	��D�Yc�A�7*

loss��M;���>       �	�TE�Yc�A�7*

loss3�	;���       �	'�E�Yc�A�7*

loss���;A�+f       �	A�F�Yc�A�7*

lossHJ^<�] z       �	�JG�Yc�A�7*

loss�K�=Q.�       �	@�G�Yc�A�7*

loss3��;
       �	�nH�Yc�A�7*

loss
�R=�աT       �	�I�Yc�A�7*

lossq-)<��V�       �	Q�I�Yc�A�7*

lossf{�<~+:�       �	WZJ�Yc�A�7*

loss�e�: �6>       �	��J�Yc�A�7*

lossC�;�v�       �	��K�Yc�A�7*

loss��t<��       �	�$L�Yc�A�7*

loss�+=���       �	(�L�Yc�A�7*

loss�<�nY{       �	p]M�Yc�A�7*

lossu0<B��{       �	��M�Yc�A�7*

loss��
;R�l�       �	8�N�Yc�A�7*

loss�ҡ;{C�       �	iO�Yc�A�7*

lossH�=��T�       �	6�O�Yc�A�7*

lossƮ;a0Q       �	DP�Yc�A�7*

loss?q�;l�(       �	��P�Yc�A�7*

loss��j<�a#�       �	�nQ�Yc�A�7*

loss�Xq<���       �	yR�Yc�A�7*

loss�>;ԧ��       �	��R�Yc�A�7*

loss=]Dk�       �	�BS�Yc�A�8*

loss�*�;y�i	       �	��S�Yc�A�8*

loss|3;D�       �	�mT�Yc�A�8*

loss=��9�:8v       �	��U�Yc�A�8*

loss?Ru:�9��       �	W
W�Yc�A�8*

loss<�&;v�p�       �	%�W�Yc�A�8*

loss�q@=#���       �	CVX�Yc�A�8*

loss8�m9��0E       �	��X�Yc�A�8*

loss�L�<���       �	1�Y�Yc�A�8*

loss*'Y:+�l�       �	�:Z�Yc�A�8*

loss(��=��T       �	=�Z�Yc�A�8*

loss,Qa<~�|�       �	��[�Yc�A�8*

loss���;(��       �	��\�Yc�A�8*

loss���<;�       �	s/]�Yc�A�8*

lossO҃;f�vp       �	��]�Yc�A�8*

loss�& ;;b��       �	�h^�Yc�A�8*

loss�i<�ԐW       �	?_�Yc�A�8*

lossjM�9��       �	�_�Yc�A�8*

loss���;2��w       �	�6`�Yc�A�8*

loss���<7� �       �	��`�Yc�A�8*

loss��(=D�	�       �	�sa�Yc�A�8*

loss�ŭ;��       �	�b�Yc�A�8*

loss��:ᕺ'       �	��b�Yc�A�8*

loss�cD<���I       �	DLc�Yc�A�8*

loss�8;RH�@       �	��c�Yc�A�8*

loss�=Ȭ�h       �	ͯd�Yc�A�8*

loss��U<��y       �	�Fe�Yc�A�8*

lossf�<c��Q       �	��e�Yc�A�8*

loss��
="UW       �	�{f�Yc�A�8*

loss}�o9�(}	       �	g�Yc�A�8*

lossÍ�;1�C�       �	��g�Yc�A�8*

loss�gg:��a       �	�Kh�Yc�A�8*

lossO�V;I��$       �	?�h�Yc�A�8*

loss1��;����