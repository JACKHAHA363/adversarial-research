       �K"	  @�Yc�Abrain.Event:2��/#�     h�k�	)!D�Yc�A"��
^
dataPlaceholder*/
_output_shapes
:���������*
dtype0*
shape: 
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
shape: *
dtype0
v
conv2d_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
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
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@*
seed2���
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
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
:@*
T0
�
conv2d_1/kernel
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
�
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
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
conv2d_1/bias/readIdentityconv2d_1/bias*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0
s
conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
s
"conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:���������@
�
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_1/ReluReluconv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
v
conv2d_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
`
conv2d_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�\1�
`
conv2d_2/random_uniform/maxConst*
valueB
 *�\1=*
dtype0*
_output_shapes
: 
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
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking(
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
[
conv2d_2/ConstConst*
dtype0*
_output_shapes
:@*
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
conv2d_2/bias/readIdentityconv2d_2/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
s
conv2d_2/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
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
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
a
dropout_1/keras_learning_phasePlaceholder*
dtype0
*
shape: *
_output_shapes
:
�
dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
:
e
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu
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
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2��
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:���������@
�
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:���������@
{
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu*
T0
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N*1
_output_shapes
:���������@: 
c
flatten_1/ShapeShapedropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
g
flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
i
flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
i
flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:
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
flatten_1/stack/0Const*
_output_shapes
: *
dtype0*
valueB :
���������
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
T0*
Tshape0*0
_output_shapes
:������������������
m
dense_1/random_uniform/shapeConst*
valueB" d  �   *
_output_shapes
:*
dtype0
_
dense_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�3z�
_
dense_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�3z<
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed���)*
T0*
dtype0*!
_output_shapes
:���*
seed2���
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
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*!
_output_shapes
:���
�
dense_1/kernel
VariableV2*
shared_name *
dtype0*
shape:���*!
_output_shapes
:���*
	container 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
~
dense_1/kernel/readIdentitydense_1/kernel*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
\
dense_1/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    
z
dense_1/bias
VariableV2*
shared_name *
dtype0*
shape:�*
_output_shapes	
:�*
	container 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:����������
]
activation_3/ReluReludense_1/BiasAdd*(
_output_shapes
:����������*
T0
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
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
_output_shapes
:*
T0

e
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*$
_class
loc:@activation_3/Relu*<
_output_shapes*
(:����������:����������*
T0

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��*
dtype0*
T0*
seed���)
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*(
_output_shapes
:����������*
T0
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
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu
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
dense_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *̈́U�
_
dense_2/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *̈́U>
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
dtype0*
seed���)*
T0*
_output_shapes
:	�
*
seed2���
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
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
dense_2/kernel/readIdentitydense_2/kernel*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
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
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

�
dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
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
valueB"         @   *
dtype0*
_output_shapes
:
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
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
:���������@*
data_formatNHWC*
T0

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
�
'sequential_1/conv2d_2/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
�
/sequential_1/conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*
strides
*
data_formatNHWC*/
_output_shapes
:���������@*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0

sequential_1/activation_2/ReluRelusequential_1/conv2d_2/BiasAdd*
T0*/
_output_shapes
:���������@
�
"sequential_1/dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
y
$sequential_1/dropout_1/cond/switch_tIdentity$sequential_1/dropout_1/cond/Switch:1*
_output_shapes
:*
T0

w
$sequential_1/dropout_1/cond/switch_fIdentity"sequential_1/dropout_1/cond/Switch*
_output_shapes
:*
T0

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
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2���
�
6sequential_1/dropout_1/cond/dropout/random_uniform/subSub6sequential_1/dropout_1/cond/dropout/random_uniform/max6sequential_1/dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/mulMul@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_1/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:���������@
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
'sequential_1/dropout_1/cond/dropout/divRealDivsequential_1/dropout_1/cond/mul-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/mulMul'sequential_1/dropout_1/cond/dropout/div)sequential_1/dropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
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
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
t
*sequential_1/flatten_1/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,sequential_1/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
f
sequential_1/flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
sequential_1/flatten_1/ProdProd$sequential_1/flatten_1/strided_slicesequential_1/flatten_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
sequential_1/flatten_1/stack/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*

axis *
_output_shapes
:*
T0*
N
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
"sequential_1/dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
y
$sequential_1/dropout_2/cond/switch_tIdentity$sequential_1/dropout_2/cond/Switch:1*
_output_shapes
:*
T0

w
$sequential_1/dropout_2/cond/switch_fIdentity"sequential_1/dropout_2/cond/Switch*
T0
*
_output_shapes
:
r
#sequential_1/dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

�
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������
�
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0
�
-sequential_1/dropout_2/cond/dropout/keep_probConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
�
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/minConst%^sequential_1/dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��7*
dtype0*
T0*
seed���)
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
6sequential_1/dropout_2/cond/dropout/random_uniform/mulMul@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
2sequential_1/dropout_2/cond/dropout/random_uniformAdd6sequential_1/dropout_2/cond/dropout/random_uniform/mul6sequential_1/dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/addAdd-sequential_1/dropout_2/cond/dropout/keep_prob2sequential_1/dropout_2/cond/dropout/random_uniform*(
_output_shapes
:����������*
T0
�
)sequential_1/dropout_2/cond/dropout/FloorFloor'sequential_1/dropout_2/cond/dropout/add*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/divRealDivsequential_1/dropout_2/cond/mul-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/mulMul'sequential_1/dropout_2/cond/dropout/div)sequential_1/dropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
�
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu*
T0
�
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul**
_output_shapes
:����������: *
N*
T0
�
sequential_1/dense_2/MatMulMatMul!sequential_1/dropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
�
sequential_1/dense_2/BiasAddBiasAddsequential_1/dense_2/MatMuldense_2/bias/read*'
_output_shapes
:���������
*
T0*
data_formatNHWC
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
shape: *
dtype0*
shared_name 
�
num_inst/AssignAssignnum_instnum_inst/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_inst
a
num_inst/readIdentitynum_inst*
T0*
_class
loc:@num_inst*
_output_shapes
: 
^
num_correct/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
o
num_correct
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
�
num_correct/AssignAssignnum_correctnum_correct/initial_value*
_class
loc:@num_correct*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
j
num_correct/readIdentitynum_correct*
T0*
_class
loc:@num_correct*
_output_shapes
: 
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
e
ArgMaxArgMaxSoftmaxArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
g
ArgMax_1ArgMaxlabelArgMax_1/dimension*#
_output_shapes
:���������*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
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
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �B
z
	AssignAdd	AssignAddnum_instConst_1*
_class
loc:@num_inst*
_output_shapes
: *
T0*
use_locking( 
~
AssignAdd_1	AssignAddnum_correctSum*
use_locking( *
T0*
_class
loc:@num_correct*
_output_shapes
: 
L
Const_2Const*
_output_shapes
: *
dtype0*
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
Const_3Const*
valueB
 *    *
_output_shapes
: *
dtype0
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
div_1/yConst*
valueB
 *  �B*
dtype0*
_output_shapes
: 
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
 softmax_cross_entropy_loss/ShapeShapediv_1*
_output_shapes
:*
out_type0*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
T0*
out_type0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
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
%softmax_cross_entropy_loss/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
T0*
_output_shapes
:
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
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
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
T0*0
_output_shapes
:������������������*
Tshape0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
T0*
_output_shapes
:*
out_type0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
T0*

axis *
N*
_output_shapes
:
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
T0*
_output_shapes
:

,softmax_cross_entropy_loss/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
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
T0*

Tidx0*
N
�
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
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
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
_output_shapes
: *
T0
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
T0*
Tshape0*#
_output_shapes
:���������
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
dtype0*
_output_shapes
: *
valueB 
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
value	B :*
_output_shapes
: *
dtype0
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
_output_shapes
:*
dtype0
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
 *    *
_output_shapes
: *
dtype0
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
�
1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
T0*
_output_shapes
: 
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
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
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
T0*
_output_shapes
: 
�
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
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
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B :
�
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:���������
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
valueB *
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
T0*
_output_shapes
: 
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
valueB *
dtype0*
_output_shapes
: 
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
�
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0
�
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
u
%softmax_cross_entropy_loss/zeros_like	ZerosLike softmax_cross_entropy_loss/Sum_1*
T0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
T0*
_output_shapes
: 
]
PlaceholderPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������

L
div_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �B
i
div_2RealDivsequential_1/dense_2/BiasAdddiv_2/y*'
_output_shapes
:���������
*
T0
c
!softmax_cross_entropy_loss_1/RankConst*
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
T0*
_output_shapes
:*
out_type0
e
#softmax_cross_entropy_loss_1/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
T0*
out_type0*
_output_shapes
:
d
"softmax_cross_entropy_loss_1/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
�
 softmax_cross_entropy_loss_1/SubSub#softmax_cross_entropy_loss_1/Rank_1"softmax_cross_entropy_loss_1/Sub/y*
_output_shapes
: *
T0
�
(softmax_cross_entropy_loss_1/Slice/beginPack softmax_cross_entropy_loss_1/Sub*
_output_shapes
:*
N*

axis *
T0
q
'softmax_cross_entropy_loss_1/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
�
"softmax_cross_entropy_loss_1/SliceSlice$softmax_cross_entropy_loss_1/Shape_1(softmax_cross_entropy_loss_1/Slice/begin'softmax_cross_entropy_loss_1/Slice/size*
Index0*
T0*
_output_shapes
:

,softmax_cross_entropy_loss_1/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
j
(softmax_cross_entropy_loss_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
�
$softmax_cross_entropy_loss_1/ReshapeReshapediv_2#softmax_cross_entropy_loss_1/concat*0
_output_shapes
:������������������*
Tshape0*
T0
e
#softmax_cross_entropy_loss_1/Rank_2Const*
value	B :*
_output_shapes
: *
dtype0
o
$softmax_cross_entropy_loss_1/Shape_2ShapePlaceholder*
_output_shapes
:*
out_type0*
T0
f
$softmax_cross_entropy_loss_1/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
"softmax_cross_entropy_loss_1/Sub_1Sub#softmax_cross_entropy_loss_1/Rank_2$softmax_cross_entropy_loss_1/Sub_1/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss_1/Slice_1/beginPack"softmax_cross_entropy_loss_1/Sub_1*
N*
T0*
_output_shapes
:*

axis 
s
)softmax_cross_entropy_loss_1/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
$softmax_cross_entropy_loss_1/Slice_1Slice$softmax_cross_entropy_loss_1/Shape_2*softmax_cross_entropy_loss_1/Slice_1/begin)softmax_cross_entropy_loss_1/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
.softmax_cross_entropy_loss_1/concat_1/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
l
*softmax_cross_entropy_loss_1/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
%softmax_cross_entropy_loss_1/concat_1ConcatV2.softmax_cross_entropy_loss_1/concat_1/values_0$softmax_cross_entropy_loss_1/Slice_1*softmax_cross_entropy_loss_1/concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*
Tshape0*0
_output_shapes
:������������������*
T0
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/Sub_2Sub!softmax_cross_entropy_loss_1/Rank$softmax_cross_entropy_loss_1/Sub_2/y*
_output_shapes
: *
T0
t
*softmax_cross_entropy_loss_1/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
)softmax_cross_entropy_loss_1/Slice_2/sizePack"softmax_cross_entropy_loss_1/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
$softmax_cross_entropy_loss_1/Slice_2Slice"softmax_cross_entropy_loss_1/Shape*softmax_cross_entropy_loss_1/Slice_2/begin)softmax_cross_entropy_loss_1/Slice_2/size*#
_output_shapes
:���������*
Index0*
T0
�
&softmax_cross_entropy_loss_1/Reshape_2Reshape%softmax_cross_entropy_loss_1/xentropy$softmax_cross_entropy_loss_1/Slice_2*
T0*#
_output_shapes
:���������*
Tshape0
~
9softmax_cross_entropy_loss_1/assert_broadcastable/weightsConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
?softmax_cross_entropy_loss_1/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :
U
Msoftmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successNoOp
�
(softmax_cross_entropy_loss_1/ToFloat_1/xConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss_1/MulMul&softmax_cross_entropy_loss_1/Reshape_2(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
"softmax_cross_entropy_loss_1/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
.softmax_cross_entropy_loss_1/num_present/EqualEqual(softmax_cross_entropy_loss_1/ToFloat_1/x0softmax_cross_entropy_loss_1/num_present/Equal/y*
T0*
_output_shapes
: 
�
3softmax_cross_entropy_loss_1/num_present/zeros_like	ZerosLike(softmax_cross_entropy_loss_1/ToFloat_1/x*
_output_shapes
: *
T0
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2softmax_cross_entropy_loss_1/num_present/ones_likeFill8softmax_cross_entropy_loss_1/num_present/ones_like/Shape8softmax_cross_entropy_loss_1/num_present/ones_like/Const*
_output_shapes
: *
T0
�
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
_output_shapes
: *
T0
�
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B :*
_output_shapes
: *
dtype0
�
ksoftmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
Dsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0
�
:softmax_cross_entropy_loss_1/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_1/num_present/SelectDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
.softmax_cross_entropy_loss_1/num_present/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss_1/Const_1ConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/Sum_1Sum softmax_cross_entropy_loss_1/Sum$softmax_cross_entropy_loss_1/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
$softmax_cross_entropy_loss_1/GreaterGreater(softmax_cross_entropy_loss_1/num_present&softmax_cross_entropy_loss_1/Greater/y*
_output_shapes
: *
T0
�
$softmax_cross_entropy_loss_1/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/EqualEqual(softmax_cross_entropy_loss_1/num_present$softmax_cross_entropy_loss_1/Equal/y*
_output_shapes
: *
T0
�
,softmax_cross_entropy_loss_1/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
,softmax_cross_entropy_loss_1/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&softmax_cross_entropy_loss_1/ones_likeFill,softmax_cross_entropy_loss_1/ones_like/Shape,softmax_cross_entropy_loss_1/ones_like/Const*
_output_shapes
: *
T0
�
#softmax_cross_entropy_loss_1/SelectSelect"softmax_cross_entropy_loss_1/Equal&softmax_cross_entropy_loss_1/ones_like(softmax_cross_entropy_loss_1/num_present*
T0*
_output_shapes
: 
�
 softmax_cross_entropy_loss_1/divRealDiv"softmax_cross_entropy_loss_1/Sum_1#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
y
'softmax_cross_entropy_loss_1/zeros_like	ZerosLike"softmax_cross_entropy_loss_1/Sum_1*
_output_shapes
: *
T0
�
"softmax_cross_entropy_loss_1/valueSelect$softmax_cross_entropy_loss_1/Greater softmax_cross_entropy_loss_1/div'softmax_cross_entropy_loss_1/zeros_like*
_output_shapes
: *
T0
P
Placeholder_1Placeholder*
_output_shapes
:*
shape: *
dtype0
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
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
�
<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like	ZerosLike softmax_cross_entropy_loss_1/div*
_output_shapes
: *
T0
�
8gradients/softmax_cross_entropy_loss_1/value_grad/SelectSelect$softmax_cross_entropy_loss_1/Greatergradients/Fill<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like*
T0*
_output_shapes
: 
�
:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1Select$softmax_cross_entropy_loss_1/Greater<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_likegradients/Fill*
T0*
_output_shapes
: 
�
Bgradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_depsNoOp9^gradients/softmax_cross_entropy_loss_1/value_grad/Select;^gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
�
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select
�
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*
_output_shapes
: *M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
x
5gradients/softmax_cross_entropy_loss_1/div_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
z
7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Egradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/div_grad/Shape7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivRealDivJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
�
3gradients/softmax_cross_entropy_loss_1/div_grad/SumSum7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivEgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0

3gradients/softmax_cross_entropy_loss_1/div_grad/NegNeg"softmax_cross_entropy_loss_1/Sum_1*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1RealDiv3gradients/softmax_cross_entropy_loss_1/div_grad/Neg#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2RealDiv9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
�
3gradients/softmax_cross_entropy_loss_1/div_grad/mulMulJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2*
_output_shapes
: *
T0
�
5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss_1/div_grad/mulGgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape
�
Jgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1*
_output_shapes
: *
T0
�
=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like	ZerosLike&softmax_cross_entropy_loss_1/ones_like*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectSelect"softmax_cross_entropy_loss_1/EqualJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like*
_output_shapes
: *
T0
�
;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1Select"softmax_cross_entropy_loss_1/Equal=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_likeJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
Cgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_depsNoOp:^gradients/softmax_cross_entropy_loss_1/Select_grad/Select<^gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1
�
Kgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependencyIdentity9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectD^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
T0*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select
�
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
_output_shapes
: 
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
�
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiplesConst*
valueB *
_output_shapes
: *
dtype0
�
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
7gradients/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
�
5gradients/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
out_type0*
_output_shapes
:*
T0
�
4gradients/softmax_cross_entropy_loss_1/Sum_grad/TileTile7gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape5gradients/softmax_cross_entropy_loss_1/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
�
=gradients/softmax_cross_entropy_loss_1/num_present_grad/ShapeShape:softmax_cross_entropy_loss_1/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
�
<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileTile?gradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape=gradients/softmax_cross_entropy_loss_1/num_present_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
out_type0*
_output_shapes
:
z
7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Egradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulMul4gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile(softmax_cross_entropy_loss_1/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/SumSum3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulEgradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Mul&softmax_cross_entropy_loss_1/Reshape_24gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile*#
_output_shapes
:���������*
T0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_1Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Ggradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_17gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
@gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape
�
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
_output_shapes
: 
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
out_type0*
_output_shapes
:*
T0
�
_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mulMul<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumSumMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1Mul/softmax_cross_entropy_loss_1/num_present/Select<gradients/softmax_cross_entropy_loss_1/num_present_grad/Tile*#
_output_shapes
:���������*
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*#
_output_shapes
:���������*
Tshape0
�
Zgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_depsNoOpR^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeT^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
bgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: 
�
dgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentitySgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*#
_output_shapes
:���������*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1*
T0
�
Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
Wgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/SumSumdgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
T0*
_output_shapes
:*
out_type0
�
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
�
gradients/zeros_like	ZerosLike'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
�
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:������������������
�
Cgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:���������*
T0
�
8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mulMul?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDimsDgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradient*
T0*0
_output_shapes
:������������������
~
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
T0*
out_type0*
_output_shapes
:
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
_
gradients/div_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/div_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_2_grad/Shapegradients/div_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/div_2_grad/RealDivRealDiv;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapediv_2/y*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/SumSumgradients/div_2_grad/RealDiv*gradients/div_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_2_grad/ReshapeReshapegradients/div_2_grad/Sumgradients/div_2_grad/Shape*
T0*'
_output_shapes
:���������
*
Tshape0
o
gradients/div_2_grad/NegNegsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
~
gradients/div_2_grad/RealDiv_1RealDivgradients/div_2_grad/Negdiv_2/y*
T0*'
_output_shapes
:���������

�
gradients/div_2_grad/RealDiv_2RealDivgradients/div_2_grad/RealDiv_1div_2/y*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/mulMul;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapegradients/div_2_grad/RealDiv_2*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/Sum_1Sumgradients/div_2_grad/mul,gradients/div_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_2_grad/Reshape_1Reshapegradients/div_2_grad/Sum_1gradients/div_2_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
m
%gradients/div_2_grad/tuple/group_depsNoOp^gradients/div_2_grad/Reshape^gradients/div_2_grad/Reshape_1
�
-gradients/div_2_grad/tuple/control_dependencyIdentitygradients/div_2_grad/Reshape&^gradients/div_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
_output_shapes
:���������

�
/gradients/div_2_grad/tuple/control_dependency_1Identitygradients/div_2_grad/Reshape_1&^gradients/div_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/div_2_grad/Reshape_1*
_output_shapes
: *
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_2_grad/tuple/control_dependency*
_output_shapes
:
*
data_formatNHWC*
T0
�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_2_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_2_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
_output_shapes
:���������

�
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
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
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	�
*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
T0
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
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_1Shapegradients/Switch:1*
_output_shapes
:*
out_type0*
T0
Z
gradients/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
t
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*(
_output_shapes
:����������
�
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros*
T0*
N**
_output_shapes
:����������: 
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0
�
Lgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_2/cond/dropout/divKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*(
_output_shapes
:����������*
Tshape0*
T0
�
Ggradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape*(
_output_shapes
:����������*
T0
�
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Lgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/NegNegsequential_1/dropout_2/cond/mul*
T0*(
_output_shapes
:����������
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Neg-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
Dgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_2/cond/mul_grad/Shape6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2gradients/sequential_1/dropout_2/cond/mul_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0
�
2gradients/sequential_1/dropout_2/cond/mul_grad/SumSum2gradients/sequential_1/dropout_2/cond/mul_grad/mulDgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6gradients/sequential_1/dropout_2/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_2/cond/mul_grad/Sum4gradients/sequential_1/dropout_2/cond/mul_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
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
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
?gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape*(
_output_shapes
:����������*
T0
�
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
T0
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_1/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*(
_output_shapes
:����������
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1**
_output_shapes
:����������: *
T0*
N
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*(
_output_shapes
:����������*
N*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad*
T0
�
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*(
_output_shapes
:����������*
T0
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
_output_shapes	
:�*
data_formatNHWC*
T0
�
<gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_3/Relu_grad/ReluGrad8^gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad*(
_output_shapes
:����������*
T0
�
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
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
T0*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*)
_output_shapes
:�����������
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*
T0
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
T0
�
Agradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
T0
�
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
_output_shapes
:*
out_type0
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
=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*1
_output_shapes
:���������@: *
T0*
N
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
out_type0*
_output_shapes
:*
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
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_1/cond/dropout/divKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1*
T0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:���������@
�
Ggradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape*
T0
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
Lgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*
T0*/
_output_shapes
:���������@*
Tshape0
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2*
T0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ggradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape
�
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
out_type0*
_output_shapes
:*
T0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Dgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_1/cond/mul_grad/Shape6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/sequential_1/dropout_1/cond/mul_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@
�
2gradients/sequential_1/dropout_1/cond/mul_grad/SumSum2gradients/sequential_1/dropout_1/cond/mul_grad/mulDgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Mul(sequential_1/dropout_1/cond/mul/Switch:1Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency*/
_output_shapes
:���������@*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1*
T0
�
gradients/Switch_3Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
c
gradients/Shape_4Shapegradients/Switch_3*
T0*
_output_shapes
:*
out_type0
\
gradients/zeros_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*/
_output_shapes
:���������@
�
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*
T0*
N*1
_output_shapes
:���������@: 
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad*/
_output_shapes
:���������@*
T0*
N
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
=gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_2/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad*/
_output_shapes
:���������@
�
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0
�
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
T0*
out_type0*
_output_shapes
:
�
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@@
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
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*&
_output_shapes
:@@*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
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
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
out_type0*
_output_shapes
:*
T0
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
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*/
_output_shapes
:���������*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput*
T0
�
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
�
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*"
_class
loc:@conv2d_1/kernel
�
beta1_power
VariableV2*
shared_name *
shape: *
_output_shapes
: *"
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
beta1_power/readIdentitybeta1_power*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
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
VariableV2*
	container *
shared_name *
dtype0*
shape: *
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
zerosConst*%
valueB@*    *&
_output_shapes
:@*
dtype0
�
conv2d_1/kernel/Adam
VariableV2*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
shape:@*
dtype0*
shared_name *
	container 
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
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
l
zeros_1Const*%
valueB@*    *
dtype0*&
_output_shapes
:@
�
conv2d_1/kernel/Adam_1
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container *
shape:@*
dtype0*&
_output_shapes
:@
�
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0
T
zeros_2Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_1/bias/Adam
VariableV2*
shared_name * 
_class
loc:@conv2d_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
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
zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_1/bias/Adam_1
VariableV2*
shared_name * 
_class
loc:@conv2d_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
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
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
l
zeros_4Const*&
_output_shapes
:@@*
dtype0*%
valueB@@*    
�
conv2d_2/kernel/Adam
VariableV2*
shape:@@*&
_output_shapes
:@@*
shared_name *"
_class
loc:@conv2d_2/kernel*
dtype0*
	container 
�
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
l
zeros_5Const*%
valueB@@*    *
dtype0*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam_1
VariableV2*
shape:@@*&
_output_shapes
:@@*
shared_name *"
_class
loc:@conv2d_2/kernel*
dtype0*
	container 
�
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking(
�
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
T
zeros_6Const*
_output_shapes
:@*
dtype0*
valueB@*    
�
conv2d_2/bias/Adam
VariableV2*
_output_shapes
:@*
dtype0*
shape:@*
	container * 
_class
loc:@conv2d_2/bias*
shared_name 
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
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
T
zeros_7Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_2/bias/Adam_1
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
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0
b
zeros_8Const* 
valueB���*    *!
_output_shapes
:���*
dtype0
�
dense_1/kernel/Adam
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
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*!
_output_shapes
:���
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
VariableV2*!
_output_shapes
:���*
dtype0*
shape:���*
	container *!
_class
loc:@dense_1/kernel*
shared_name 
�
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1zeros_9*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
�
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0
W
zeros_10Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_1/bias/Adam
VariableV2*
	container *
dtype0*
_class
loc:@dense_1/bias*
shared_name *
_output_shapes	
:�*
shape:�
�
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
|
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
W
zeros_11Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dense_1/bias/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:�*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
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
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*
_output_shapes
:	�
*
validate_shape(*!
_class
loc:@dense_2/kernel*
T0*
use_locking(
�
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
T0
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
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1zeros_13*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�

�
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
T0
U
zeros_14Const*
dtype0*
_output_shapes
:
*
valueB
*    
�
dense_2/bias/Adam
VariableV2*
shared_name *
_class
loc:@dense_2/bias*
	container *
shape:
*
dtype0*
_output_shapes
:

�
dense_2/bias/Adam/AssignAssigndense_2/bias/Adamzeros_14*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
{
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0
U
zeros_15Const*
valueB
*    *
dtype0*
_output_shapes
:

�
dense_2/bias/Adam_1
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *
_class
loc:@dense_2/bias*
shared_name 
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

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
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
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

�
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
 Bloss*
_output_shapes
: *
dtype0
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "��u�,     ��#�	��F�Yc�AJ��
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
dataPlaceholder*/
_output_shapes
:���������*
dtype0*
shape: 
W
labelPlaceholder*
shape: *
dtype0*'
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
 *�x�*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�x=
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2���*
T0*
seed���)*
dtype0
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0
�
conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
:@
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
conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
[
conv2d_1/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
y
conv2d_1/bias
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
�
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
s
conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:���������@
�
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:���������@
e
activation_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
v
conv2d_2/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
valueB
 *�\1�*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�\1=
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2���*
dtype0*
T0*
seed���)
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
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking(
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
[
conv2d_2/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
y
conv2d_2/bias
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
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
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
s
conv2d_2/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������@*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC
e
activation_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
a
dropout_1/keras_learning_phasePlaceholder*
dtype0
*
shape: *
_output_shapes
:
�
dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
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
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
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
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2��
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*/
_output_shapes
:���������@*
T0
�
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*/
_output_shapes
:���������@*
T0
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
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N*1
_output_shapes
:���������@: 
c
flatten_1/ShapeShapedropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
g
flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
i
flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
end_mask*
ellipsis_mask *

begin_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
T0*
Index0
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
flatten_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
���������
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
N*
T0*
_output_shapes
:*

axis 
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*0
_output_shapes
:������������������*
Tshape0*
T0
m
dense_1/random_uniform/shapeConst*
valueB" d  �   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�3z�*
dtype0*
_output_shapes
: 
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
seed2���*
dtype0*
T0*
seed���)
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*!
_output_shapes
:���
�
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*!
_output_shapes
:���
�
dense_1/kernel
VariableV2*
shared_name *
dtype0*
shape:���*!
_output_shapes
:���*
	container 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
~
dense_1/kernel/readIdentitydense_1/kernel*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
\
dense_1/ConstConst*
_output_shapes	
:�*
dtype0*
valueB�*    
z
dense_1/bias
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
shape:�
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
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:����������
]
activation_3/ReluReludense_1/BiasAdd*(
_output_shapes
:����������*
T0
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
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
_output_shapes
:*
T0

e
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
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
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2��
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
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*(
_output_shapes
:����������*
T0
t
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
�
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*
T0*$
_class
loc:@activation_3/Relu*<
_output_shapes*
(:����������:����������
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
dtype0*
seed���)*
T0*
_output_shapes
:	�
*
seed2���
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes
:	�


dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	�

�
dense_2/kernel
VariableV2*
_output_shapes
:	�
*
	container *
dtype0*
shared_name *
shape:	�

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
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

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
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

�
dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*'
_output_shapes
:���������
*
T0*
data_formatNHWC
�
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
�
'sequential_1/conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
�
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0
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
"sequential_1/dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
y
$sequential_1/dropout_1/cond/switch_tIdentity$sequential_1/dropout_1/cond/Switch:1*
_output_shapes
:*
T0

w
$sequential_1/dropout_1/cond/switch_fIdentity"sequential_1/dropout_1/cond/Switch*
_output_shapes
:*
T0

r
#sequential_1/dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
�
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
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
 *  @?*
dtype0*
_output_shapes
: 
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
�
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2���*
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
'sequential_1/dropout_1/cond/dropout/mulMul'sequential_1/dropout_1/cond/dropout/div)sequential_1/dropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
�
$sequential_1/dropout_1/cond/Switch_1Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu
�
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*
T0*
N*1
_output_shapes
:���������@: 
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
t
*sequential_1/flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
v
,sequential_1/flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
:*
end_mask*
Index0*
T0*
shrink_axis_mask *
new_axis_mask 
f
sequential_1/flatten_1/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
sequential_1/flatten_1/ReshapeReshape!sequential_1/dropout_1/cond/Mergesequential_1/flatten_1/stack*
Tshape0*0
_output_shapes
:������������������*
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
#sequential_1/dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

�
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������*
T0
�
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
�
-sequential_1/dropout_2/cond/dropout/keep_probConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
�
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/minConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
dtype0*
seed���)*
T0*(
_output_shapes
:����������*
seed2��7
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
6sequential_1/dropout_2/cond/dropout/random_uniform/mulMul@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
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
)sequential_1/dropout_2/cond/dropout/FloorFloor'sequential_1/dropout_2/cond/dropout/add*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/divRealDivsequential_1/dropout_2/cond/mul-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/mulMul'sequential_1/dropout_2/cond/dropout/div)sequential_1/dropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
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
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
�
sequential_1/dense_2/BiasAddBiasAddsequential_1/dense_2/MatMuldense_2/bias/read*'
_output_shapes
:���������
*
data_formatNHWC*
T0
b
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

[
num_inst/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
num_inst
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
�
num_inst/AssignAssignnum_instnum_inst/initial_value*
_class
loc:@num_inst*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
a
num_inst/readIdentitynum_inst*
_class
loc:@num_inst*
_output_shapes
: *
T0
^
num_correct/initial_valueConst*
dtype0*
_output_shapes
: *
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
num_correct/AssignAssignnum_correctnum_correct/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@num_correct*
T0*
use_locking(
j
num_correct/readIdentitynum_correct*
T0*
_output_shapes
: *
_class
loc:@num_correct
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
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
	AssignAdd	AssignAddnum_instConst_1*
_class
loc:@num_inst*
_output_shapes
: *
T0*
use_locking( 
~
AssignAdd_1	AssignAddnum_correctSum*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@num_correct
L
Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
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
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.
A
addAddnum_inst/readadd/y*
_output_shapes
: *
T0
F
divRealDivnum_correct/readadd*
_output_shapes
: *
T0
L
div_1/yConst*
valueB
 *  �B*
dtype0*
_output_shapes
: 
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*'
_output_shapes
:���������
*
T0
a
softmax_cross_entropy_loss/RankConst*
value	B :*
_output_shapes
: *
dtype0
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
_output_shapes
:*
out_type0*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
_output_shapes
:*
out_type0*
T0
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
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
T0*

axis *
N*
_output_shapes
:
o
%softmax_cross_entropy_loss/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
_output_shapes
:*
Index0*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
Tshape0*0
_output_shapes
:������������������*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
T0*
_output_shapes
:*
out_type0
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
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
T0*
_output_shapes
:
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
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
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
(softmax_cross_entropy_loss/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
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
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
dtype0*
_output_shapes
: *
valueB 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
_output_shapes
:*
out_type0*
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
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:���������*
T0
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
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
�
1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
T0*
_output_shapes
: 
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
�
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
_output_shapes
:*
dtype0
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
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
T0*
_output_shapes
: 
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
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
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
T0*
_output_shapes
: 
]
PlaceholderPlaceholder*'
_output_shapes
:���������
*
dtype0*
shape: 
L
div_2/yConst*
valueB
 *  �B*
dtype0*
_output_shapes
: 
i
div_2RealDivsequential_1/dense_2/BiasAdddiv_2/y*'
_output_shapes
:���������
*
T0
c
!softmax_cross_entropy_loss_1/RankConst*
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
T0*
_output_shapes
:*
out_type0
e
#softmax_cross_entropy_loss_1/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
T0*
_output_shapes
:*
out_type0
d
"softmax_cross_entropy_loss_1/Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
�
 softmax_cross_entropy_loss_1/SubSub#softmax_cross_entropy_loss_1/Rank_1"softmax_cross_entropy_loss_1/Sub/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss_1/Slice/beginPack softmax_cross_entropy_loss_1/Sub*
T0*

axis *
N*
_output_shapes
:
q
'softmax_cross_entropy_loss_1/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
"softmax_cross_entropy_loss_1/SliceSlice$softmax_cross_entropy_loss_1/Shape_1(softmax_cross_entropy_loss_1/Slice/begin'softmax_cross_entropy_loss_1/Slice/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss_1/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
j
(softmax_cross_entropy_loss_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
$softmax_cross_entropy_loss_1/ReshapeReshapediv_2#softmax_cross_entropy_loss_1/concat*0
_output_shapes
:������������������*
Tshape0*
T0
e
#softmax_cross_entropy_loss_1/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
o
$softmax_cross_entropy_loss_1/Shape_2ShapePlaceholder*
T0*
_output_shapes
:*
out_type0
f
$softmax_cross_entropy_loss_1/Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss_1/Sub_1Sub#softmax_cross_entropy_loss_1/Rank_2$softmax_cross_entropy_loss_1/Sub_1/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss_1/Slice_1/beginPack"softmax_cross_entropy_loss_1/Sub_1*

axis *
_output_shapes
:*
T0*
N
s
)softmax_cross_entropy_loss_1/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
$softmax_cross_entropy_loss_1/Slice_1Slice$softmax_cross_entropy_loss_1/Shape_2*softmax_cross_entropy_loss_1/Slice_1/begin)softmax_cross_entropy_loss_1/Slice_1/size*
_output_shapes
:*
Index0*
T0
�
.softmax_cross_entropy_loss_1/concat_1/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
l
*softmax_cross_entropy_loss_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
%softmax_cross_entropy_loss_1/concat_1ConcatV2.softmax_cross_entropy_loss_1/concat_1/values_0$softmax_cross_entropy_loss_1/Slice_1*softmax_cross_entropy_loss_1/concat_1/axis*
_output_shapes
:*
T0*

Tidx0*
N
�
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/Sub_2Sub!softmax_cross_entropy_loss_1/Rank$softmax_cross_entropy_loss_1/Sub_2/y*
_output_shapes
: *
T0
t
*softmax_cross_entropy_loss_1/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
�
)softmax_cross_entropy_loss_1/Slice_2/sizePack"softmax_cross_entropy_loss_1/Sub_2*

axis *
_output_shapes
:*
T0*
N
�
$softmax_cross_entropy_loss_1/Slice_2Slice"softmax_cross_entropy_loss_1/Shape*softmax_cross_entropy_loss_1/Slice_2/begin)softmax_cross_entropy_loss_1/Slice_2/size*#
_output_shapes
:���������*
Index0*
T0
�
&softmax_cross_entropy_loss_1/Reshape_2Reshape%softmax_cross_entropy_loss_1/xentropy$softmax_cross_entropy_loss_1/Slice_2*
T0*#
_output_shapes
:���������*
Tshape0
~
9softmax_cross_entropy_loss_1/assert_broadcastable/weightsConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
?softmax_cross_entropy_loss_1/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
out_type0*
_output_shapes
:*
T0

=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
U
Msoftmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successNoOp
�
(softmax_cross_entropy_loss_1/ToFloat_1/xConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
 softmax_cross_entropy_loss_1/MulMul&softmax_cross_entropy_loss_1/Reshape_2(softmax_cross_entropy_loss_1/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
"softmax_cross_entropy_loss_1/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
valueB: 
�
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
.softmax_cross_entropy_loss_1/num_present/EqualEqual(softmax_cross_entropy_loss_1/ToFloat_1/x0softmax_cross_entropy_loss_1/num_present/Equal/y*
_output_shapes
: *
T0
�
3softmax_cross_entropy_loss_1/num_present/zeros_like	ZerosLike(softmax_cross_entropy_loss_1/ToFloat_1/x*
_output_shapes
: *
T0
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2softmax_cross_entropy_loss_1/num_present/ones_likeFill8softmax_cross_entropy_loss_1/num_present/ones_like/Shape8softmax_cross_entropy_loss_1/num_present/ones_like/Const*
_output_shapes
: *
T0
�
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
T0*
_output_shapes
: 
�
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B : 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B :*
_output_shapes
: *
dtype0
�
ksoftmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
Dsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0
�
:softmax_cross_entropy_loss_1/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_1/num_present/SelectDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
.softmax_cross_entropy_loss_1/num_present/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
$softmax_cross_entropy_loss_1/Const_1ConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
"softmax_cross_entropy_loss_1/Sum_1Sum softmax_cross_entropy_loss_1/Sum$softmax_cross_entropy_loss_1/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
$softmax_cross_entropy_loss_1/GreaterGreater(softmax_cross_entropy_loss_1/num_present&softmax_cross_entropy_loss_1/Greater/y*
_output_shapes
: *
T0
�
$softmax_cross_entropy_loss_1/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
"softmax_cross_entropy_loss_1/EqualEqual(softmax_cross_entropy_loss_1/num_present$softmax_cross_entropy_loss_1/Equal/y*
_output_shapes
: *
T0
�
,softmax_cross_entropy_loss_1/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
,softmax_cross_entropy_loss_1/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&softmax_cross_entropy_loss_1/ones_likeFill,softmax_cross_entropy_loss_1/ones_like/Shape,softmax_cross_entropy_loss_1/ones_like/Const*
T0*
_output_shapes
: 
�
#softmax_cross_entropy_loss_1/SelectSelect"softmax_cross_entropy_loss_1/Equal&softmax_cross_entropy_loss_1/ones_like(softmax_cross_entropy_loss_1/num_present*
T0*
_output_shapes
: 
�
 softmax_cross_entropy_loss_1/divRealDiv"softmax_cross_entropy_loss_1/Sum_1#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
y
'softmax_cross_entropy_loss_1/zeros_like	ZerosLike"softmax_cross_entropy_loss_1/Sum_1*
_output_shapes
: *
T0
�
"softmax_cross_entropy_loss_1/valueSelect$softmax_cross_entropy_loss_1/Greater softmax_cross_entropy_loss_1/div'softmax_cross_entropy_loss_1/zeros_like*
T0*
_output_shapes
: 
P
Placeholder_1Placeholder*
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
gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
�
<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like	ZerosLike softmax_cross_entropy_loss_1/div*
T0*
_output_shapes
: 
�
8gradients/softmax_cross_entropy_loss_1/value_grad/SelectSelect$softmax_cross_entropy_loss_1/Greatergradients/Fill<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like*
T0*
_output_shapes
: 
�
:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1Select$softmax_cross_entropy_loss_1/Greater<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
�
Bgradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_depsNoOp9^gradients/softmax_cross_entropy_loss_1/value_grad/Select;^gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
�
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select*
_output_shapes
: *
T0
�
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*
_output_shapes
: *M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
x
5gradients/softmax_cross_entropy_loss_1/div_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
z
7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Egradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/div_grad/Shape7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivRealDivJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
�
3gradients/softmax_cross_entropy_loss_1/div_grad/SumSum7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivEgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
T0*
_output_shapes
: *
Tshape0

3gradients/softmax_cross_entropy_loss_1/div_grad/NegNeg"softmax_cross_entropy_loss_1/Sum_1*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1RealDiv3gradients/softmax_cross_entropy_loss_1/div_grad/Neg#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2RealDiv9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
�
3gradients/softmax_cross_entropy_loss_1/div_grad/mulMulJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss_1/div_grad/mulGgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape
�
Jgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
T0*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1
�
=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like	ZerosLike&softmax_cross_entropy_loss_1/ones_like*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectSelect"softmax_cross_entropy_loss_1/EqualJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1Select"softmax_cross_entropy_loss_1/Equal=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_likeJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Cgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_depsNoOp:^gradients/softmax_cross_entropy_loss_1/Select_grad/Select<^gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1
�
Kgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependencyIdentity9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectD^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select*
_output_shapes
: 
�
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
_output_shapes
: *
T0
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 
�
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
�
5gradients/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
T0*
out_type0*
_output_shapes
:
�
4gradients/softmax_cross_entropy_loss_1/Sum_grad/TileTile7gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape5gradients/softmax_cross_entropy_loss_1/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
�
=gradients/softmax_cross_entropy_loss_1/num_present_grad/ShapeShape:softmax_cross_entropy_loss_1/num_present/broadcast_weights*
_output_shapes
:*
out_type0*
T0
�
<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileTile?gradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape=gradients/softmax_cross_entropy_loss_1/num_present_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2*
_output_shapes
:*
out_type0*
T0
z
7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Egradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulMul4gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile(softmax_cross_entropy_loss_1/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/SumSum3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulEgradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Mul&softmax_cross_entropy_loss_1/Reshape_24gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile*
T0*#
_output_shapes
:���������
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_1Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Ggradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_17gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
@gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape
�
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:*
out_type0
�
_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mulMul<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumSumMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1Mul/softmax_cross_entropy_loss_1/num_present/Select<gradients/softmax_cross_entropy_loss_1/num_present_grad/Tile*#
_output_shapes
:���������*
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
�
Zgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_depsNoOpR^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeT^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
bgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*
_output_shapes
: *d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape*
T0
�
dgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentitySgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������*
T0
�
Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
Wgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/SumSumdgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
out_type0*
_output_shapes
:*
T0
�
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
�
gradients/zeros_like	ZerosLike'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:������������������
�
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:������������������
�
Cgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mulMul?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDimsDgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
~
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
T0*
out_type0*
_output_shapes
:
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
_
gradients/div_2_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
*gradients/div_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_2_grad/Shapegradients/div_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/div_2_grad/RealDivRealDiv;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapediv_2/y*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/SumSumgradients/div_2_grad/RealDiv*gradients/div_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_2_grad/ReshapeReshapegradients/div_2_grad/Sumgradients/div_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

o
gradients/div_2_grad/NegNegsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

~
gradients/div_2_grad/RealDiv_1RealDivgradients/div_2_grad/Negdiv_2/y*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/RealDiv_2RealDivgradients/div_2_grad/RealDiv_1div_2/y*
T0*'
_output_shapes
:���������

�
gradients/div_2_grad/mulMul;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapegradients/div_2_grad/RealDiv_2*
T0*'
_output_shapes
:���������

�
gradients/div_2_grad/Sum_1Sumgradients/div_2_grad/mul,gradients/div_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_2_grad/Reshape_1Reshapegradients/div_2_grad/Sum_1gradients/div_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/div_2_grad/tuple/group_depsNoOp^gradients/div_2_grad/Reshape^gradients/div_2_grad/Reshape_1
�
-gradients/div_2_grad/tuple/control_dependencyIdentitygradients/div_2_grad/Reshape&^gradients/div_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
_output_shapes
:���������
*
T0
�
/gradients/div_2_grad/tuple/control_dependency_1Identitygradients/div_2_grad/Reshape_1&^gradients/div_2_grad/tuple/group_deps*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients/div_2_grad/Reshape_1
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_2_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:

�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_2_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_2_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_2_grad/Reshape
�
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:
*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
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
T0*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	�

�
:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradSwitchCgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency#sequential_1/dropout_2/cond/pred_id*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*<
_output_shapes*
(:����������:����������*
T0
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
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
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
 *    *
_output_shapes
: *
dtype0
t
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*(
_output_shapes
:����������*
T0
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
Lgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
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
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
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
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2*(
_output_shapes
:����������*
T0
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
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape*
T0
�
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
out_type0*
_output_shapes
:*
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
2gradients/sequential_1/dropout_2/cond/mul_grad/SumSum2gradients/sequential_1/dropout_2/cond/mul_grad/mulDgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6gradients/sequential_1/dropout_2/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_2/cond/mul_grad/Sum4gradients/sequential_1/dropout_2/cond/mul_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Mul(sequential_1/dropout_2/cond/mul/Switch:1Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
?gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape
�
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
_output_shapes
: *
T0
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_2Shapegradients/Switch_1*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_1/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*(
_output_shapes
:����������
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1*
N*
T0**
_output_shapes
:����������: 
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*
T0*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad*
N*(
_output_shapes
:����������
�
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*(
_output_shapes
:����������*
T0
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
_output_shapes	
:�*
T0*
data_formatNHWC
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
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*)
_output_shapes
:�����������*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*
T0
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*
T0
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
T0*/
_output_shapes
:���������@*
Tshape0
�
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
T0
�
Agradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
T0
�
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
_output_shapes
:*
out_type0
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
=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*1
_output_shapes
:���������@: *
N*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
_output_shapes
:*
out_type0*
T0
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
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
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
T0*/
_output_shapes
:���������@*
Tshape0
�
Ggradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape*
T0
�
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:���������@*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
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
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/NegNegsequential_1/dropout_1/cond/mul*/
_output_shapes
:���������@*
T0
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
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
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
T0*
out_type0*
_output_shapes
:
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
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
2gradients/sequential_1/dropout_1/cond/mul_grad/SumSum2gradients/sequential_1/dropout_1/cond/mul_grad/mulDgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Mul(sequential_1/dropout_1/cond/mul/Switch:1Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency*
T0*/
_output_shapes
:���������@
�
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
?gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_1/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape*/
_output_shapes
:���������@
�
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1*
_output_shapes
: 
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
T0*
N*1
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
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad*/
_output_shapes
:���������@
�
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad
�
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
_output_shapes
:*
out_type0*
T0
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
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*%
valueB"      @   @   *
_output_shapes
:*
dtype0
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*&
_output_shapes
:@@*
paddingVALID*
T0*
use_cudnn_on_gpu(
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
T0*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
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
T0*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
T0*
_output_shapes
:*
out_type0
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
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         @   
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
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
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*&
_output_shapes
:@*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0
�
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*"
_class
loc:@conv2d_1/kernel
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
n
beta1_power/readIdentitybeta1_power*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
�
beta2_power/initial_valueConst*
valueB
 *w�?*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
_output_shapes
: *
dtype0*
shape: *
	container *"
_class
loc:@conv2d_1/kernel*
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
n
beta2_power/readIdentitybeta2_power*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
j
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    
�
conv2d_1/kernel/Adam
VariableV2*
shape:@*&
_output_shapes
:@*
shared_name *"
_class
loc:@conv2d_1/kernel*
dtype0*
	container 
�
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
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
VariableV2*
shape:@*&
_output_shapes
:@*
shared_name *"
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
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0
T
zeros_3Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_1/bias/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
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
zeros_4Const*%
valueB@@*    *
dtype0*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam
VariableV2*
shared_name *"
_class
loc:@conv2d_2/kernel*
	container *
shape:@@*
dtype0*&
_output_shapes
:@@
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
zeros_5Const*%
valueB@@*    *
dtype0*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam_1
VariableV2*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
shape:@@*
dtype0*
shared_name *
	container 
�
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
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
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
~
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0
T
zeros_7Const*
valueB@*    *
dtype0*
_output_shapes
:@
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
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
b
zeros_8Const* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
dense_1/kernel/Adam
VariableV2*!
_output_shapes
:���*
dtype0*
shape:���*
	container *!
_class
loc:@dense_1/kernel*
shared_name 
�
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*!
_output_shapes
:���
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
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
loc:@dense_1/kernel*
shared_name *!
_output_shapes
:���*
shape:���
�
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1zeros_9*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
�
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
W
zeros_10Const*
_output_shapes	
:�*
dtype0*
valueB�*    
�
dense_1/bias/Adam
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
shape:�*
dtype0*
shared_name *
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
zeros_11Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_1/bias/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@dense_1/bias*
shared_name *
_output_shapes	
:�*
shape:�
�
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
�
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
_
zeros_12Const*
valueB	�
*    *
_output_shapes
:	�
*
dtype0
�
dense_2/kernel/Adam
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
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�

�
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
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
VariableV2*
_output_shapes
:	�
*
dtype0*
shape:	�
*
	container *!
_class
loc:@dense_2/kernel*
shared_name 
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
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
U
zeros_14Const*
valueB
*    *
dtype0*
_output_shapes
:

�
dense_2/bias/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:
*
_output_shapes
:
*
_class
loc:@dense_2/bias
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
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0*
use_locking( 
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
use_locking( 
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
T0*
use_locking( 
�
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
�
Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
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
	loss/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N""V
lossesL
J
"softmax_cross_entropy_loss/value:0
$softmax_cross_entropy_loss_1/value:0"
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0WG{�       ��-	�t�Yc�A*

lossf`@p䩁       ��-	��t�Yc�A*

loss};@���b       ��-	��u�Yc�A*

loss�@^��|       ��-	kav�Yc�A*

loss�@�3�G       ��-	�w�Yc�A*

lossWz@F�2�       ��-	b�w�Yc�A*

loss��@<��n       ��-	�^x�Yc�A*

lossX�@u��       ��-	ܡy�Yc�A*

loss��@uZ��       ��-	�Xz�Yc�A	*

lossHE
@7�:�       ��-	{�Yc�A
*

loss�C@L@�       ��-	'�{�Yc�A*

loss��?�H��       ��-	�s|�Yc�A*

loss#s�?�3Q       ��-	C}�Yc�A*

loss��?��+0       ��-	-�}�Yc�A*

lossWO�?/~@�       ��-	fL~�Yc�A*

loss�d�?�l       ��-	 �~�Yc�A*

loss8�?�_K       ��-	9{�Yc�A*

loss�T�?��}       ��-	���Yc�A*

lossq��?c�`�       ��-	ŭ��Yc�A*

loss7��?�f�E       ��-	O��Yc�A*

loss��?C�?       ��-	큾Yc�A*

lossH��?^���       ��-	J��Yc�A*

loss�x�?x7�       ��-	w��Yc�A*

loss���?DkZ�       ��-	㥃�Yc�A*

loss��?���       ��-	P5��Yc�A*

loss���?���       ��-	�Ƅ�Yc�A*

loss2Mp?1��       ��-	W��Yc�A*

loss]N�?�v��       ��-	셾Yc�A*

loss�;�?Z���       ��-	���Yc�A*

loss��a?�:`�       ��-	X��Yc�A*

loss��?!z�       ��-	d���Yc�A*

lossm"?$z\1       ��-	EF��Yc�A *

lossS�u?�&�       ��-	�䈾Yc�A!*

loss�k{?�<�       ��-	����Yc�A"*

loss{$g?���W       ��-	)?��Yc�A#*

lossh{T?�n�       ��-	a8��Yc�A$*

loss,?Y_��       ��-	�ҋ�Yc�A%*

lossh�?��J.       ��-	�r��Yc�A&*

loss1��?q���       ��-	�e��Yc�A'*

loss��?�ڝe       ��-	C��Yc�A(*

loss��0?�s[�       ��-	���Yc�A)*

loss�7?	I$       ��-	CS��Yc�A**

loss��>?��       ��-	�ꏾYc�A+*

lossHC=?�0�'       ��-	����Yc�A,*

loss�B?ܚ�w       ��-	�+��Yc�A-*

loss|�4?�yA<       ��-	~���Yc�A.*

loss� H?���       ��-	䃒�Yc�A/*

loss{+'?�Ɓ       ��-	\ ��Yc�A0*

loss��%?VD��       ��-	G˓�Yc�A1*

loss!�?��%�       ��-	�i��Yc�A2*

loss��?Mh�       ��-	K��Yc�A3*

lossn�?E��       ��-	����Yc�A4*

loss�?�X        ��-	�@��Yc�A5*

loss��B?\G�       ��-	`��Yc�A6*

lossM{�>�Ƥ�       ��-	h���Yc�A7*

loss�"?���       ��-	2W��Yc�A8*

losss<�>9��       ��-	��Yc�A9*

loss��>z���       ��-	}���Yc�A:*

loss�E?��       ��-	�6��Yc�A;*

loss7��>���y       ��-	-Ϛ�Yc�A<*

lossW�9?��k       ��-	kd��Yc�A=*

lossV�?̪�{       ��-	���Yc�A>*

loss�3?����       ��-	ė��Yc�A?*

loss�a?��n       ��-	~;��Yc�A@*

loss���>��[�       ��-	�՝�Yc�AA*

loss&��>�@�-       ��-	fl��Yc�AB*

loss�we?���       ��-	N��Yc�AC*

loss[�5?= 96       ��-	i���Yc�AD*

lossd�7?�ƦE       ��-	=
��Yc�AE*

lossʄ?�ޞ       ��-	6���Yc�AF*

loss��?G�a_       ��-	*T��Yc�AG*

loss�(�>���
       ��-	����Yc�AH*

loss�#?Mk)�       ��-	�C��Yc�AI*

lossU,?�C/       ��-	�<��Yc�AJ*

lossE�Z?��       ��-	�ץ�Yc�AK*

loss!(:?�La2       ��-	Gt��Yc�AL*

losse�M?=_k�       ��-	���Yc�AM*

loss W?���       ��-	���Yc�AN*

loss��(? ߉k       ��-	�O��Yc�AO*

lossR�> �^5       ��-	���Yc�AP*

loss��I?��ʍ       ��-	Q���Yc�AQ*

loss%�z?t	        ��-	-���Yc�AR*

loss/A?�oJF       ��-	[��Yc�AS*

loss}��>wdU�       ��-	N��Yc�AT*

loss�?�-�       ��-	����Yc�AU*

loss���>�	A       ��-	����Yc�AV*

loss��>9�H       ��-	hZ��Yc�AW*

loss��?����       ��-	����Yc�AX*

loss��?8n       ��-	����Yc�AY*

loss�L?�Ք�       ��-	xC��Yc�AZ*

loss�K?Gl�       ��-	�갾Yc�A[*

lossM�?P�W       ��-	����Yc�A\*

loss��%?��       ��-	K:��Yc�A]*

losst7
?�FL�       ��-	/ܲ�Yc�A^*

loss�C4?C��       ��-	�w��Yc�A_*

loss|��>0���       ��-	fO��Yc�A`*

lossË?Ϥ�       ��-	u��Yc�Aa*

loss�SS?B�v�       ��-	�ȵ�Yc�Ab*

loss��?�@H�       ��-	/ᶾYc�Ac*

lossx?���       ��-	���Yc�Ad*

loss�-�>�!��       ��-	>w��Yc�Ae*

loss;��>4(�.       ��-	��Yc�Af*

loss!�!?;YC       ��-	���Yc�Ag*

loss�q�>2�?�       ��-	����Yc�Ah*

loss-��>u,�       ��-	�Q��Yc�Ai*

loss���>��_�       ��-	y껾Yc�Aj*

loss�.?� �u       ��-	o���Yc�Ak*

loss��>�B       ��-	��Yc�Al*

loss��O?&r��       ��-	c���Yc�Am*

loss�b?��`�       ��-	�J��Yc�An*

loss$�?W��Y       ��-	�꾾Yc�Ao*

lossL$?�l��       ��-	���Yc�Ap*

loss=��>���u       ��-	���Yc�Aq*

loss�ɹ>��`       ��-	c���Yc�Ar*

loss���>(��       ��-	bM��Yc�As*

lossH��>�Y       ��-	���Yc�At*

loss���>�mt8       ��-	o~¾Yc�Au*

loss��?��       ��-	+þYc�Av*

loss|?
۱`       ��-	��þYc�Aw*

losse�>2M|       ��-	NľYc�Ax*

loss�4�>Oܜg       ��-	o�žYc�Ay*

loss���>���       ��-	�nƾYc�Az*

loss���>f�s       ��-	�ǾYc�A{*

loss�
?!�F       ��-	ȾYc�A|*

loss�v�>|�       ��-	[�ȾYc�A}*

loss\� ?��F�       ��-	�\ɾYc�A~*

loss��?���       ��-	��ɾYc�A*

lossqI
?��X       �	y�ʾYc�A�*

loss�,�>U�Kj       �	�2˾Yc�A�*

loss}c�>�:9       �	\�˾Yc�A�*

loss �>#��,       �	�\̾Yc�A�*

loss>z�>E㜎       �	��̾Yc�A�*

lossagk>�7       �	�;Yc�A�*

loss]�>��       �	�GξYc�A�*

lossD�>w��       �	�ξYc�A�*

lossm�?����       �	�ϾYc�A�*

loss���>�|�t       �	O#оYc�A�*

lossG�>I���       �	�оYc�A�*

loss�3�>
~+M       �	�\ѾYc�A�*

loss?z�>A�{`       �	�ҾYc�A�*

loss�o>;�g       �	�ҾYc�A�*

loss��?�T@       �	X;ӾYc�A�*

lossqز>����       �	��ӾYc�A�*

loss^�>�_��       �	epԾYc�A�*

loss=P�>a�Y       �	վYc�A�*

loss���>��        �	�վYc�A�*

lossN�>EU�s       �	,I־Yc�A�*

loss�Y]>�d��       �	��־Yc�A�*

loss{��>/�O�       �	��׾Yc�A�*

loss�"�>m;	i       �	�,ؾYc�A�*

loss�N�>V�$g       �	��ؾYc�A�*

loss�?��3�       �	�kپYc�A�*

lossv��>��Ԣ       �	�ھYc�A�*

loss�"�>�9�9       �	ۤھYc�A�*

lossf��>Z�x        �	[@۾Yc�A�*

loss�]a>W��&       �	��۾Yc�A�*

loss��>��       �	oܾYc�A�*

loss�ϼ>d!^7       �	�&ݾYc�A�*

loss�;?؍!�       �	��ݾYc�A�*

loss�C�>�C�       �	�_޾Yc�A�*

loss]�?�s.D       �	��޾Yc�A�*

loss���>
~       �	�߾Yc�A�*

lossߺ�>�{��       �	�2�Yc�A�*

lossh�>30�C       �	���Yc�A�*

loss�H?���       �	^�Yc�A�*

loss��>�lc�       �	D��Yc�A�*

loss��>G�       �	���Yc�A�*

lossѽ�>+��e       �	�7�Yc�A�*

loss-��>��_�       �	���Yc�A�*

loss��>0:�H       �	lw�Yc�A�*

loss�.}>��Y\       �	��Yc�A�*

loss�Aa>�,�"       �	��Yc�A�*

loss1t?�7       �	�s�Yc�A�*

lossJX?���q       �	��Yc�A�*

loss�΍>��1�       �	/��Yc�A�*

lossB�>}U�       �	|��Yc�A�*

lossȻ@?��       �	7�Yc�A�*

loss�2�>�\��       �	���Yc�A�*

loss��>�3�0       �	��Yc�A�*

loss�F�>�g        �	�1�Yc�A�*

loss�T>A��       �	���Yc�A�*

loss��>{�        �	���Yc�A�*

loss�U�>$%�{       �	)%��Yc�A�*

loss���>��֋       �	����Yc�A�*

loss]�>�B�       �	�l�Yc�A�*

loss��>�|�I       �	v�Yc�A�*

loss��>qϧ�       �	5��Yc�A�*

loss���>���g       �	zP�Yc�A�*

loss��>��#�       �	��Yc�A�*

loss���>�+<�       �	���Yc�A�*

lossRM�>L�'�       �	f0�Yc�A�*

loss35�>�fP^       �	"��Yc�A�*

loss�Ǖ>�ڍ�       �	b�Yc�A�*

loss
�>�%ģ       �	m��Yc�A�*

lossD�h>3?��       �	.���Yc�A�*

lossF��>=2m�       �	YM��Yc�A�*

loss�>��[       �	W���Yc�A�*

loss���>�?I       �	ގ��Yc�A�*

loss]�>��2       �	YN��Yc�A�*

loss�̉>\N�       �	����Yc�A�*

lossO6�>A@�       �	j���Yc�A�*

loss��w><�v       �	�?��Yc�A�*

lossx,�>�q�j       �	���Yc�A�*

loss���>�G��       �	����Yc�A�*

loss3U�>�ܦd       �	�6��Yc�A�*

loss�&>.O��       �	����Yc�A�*

lossgl>��|       �	e���Yc�A�*

lossj�>��%�       �	�L��Yc�A�*

loss�\>��       �	\���Yc�A�*

loss��>�N��       �	^���Yc�A�*

loss-b�>h��       �	���Yc�A�*

loss��>kS��       �	���Yc�A�*

loss��.>�M�       �	4J �Yc�A�*

loss�3�>ѹ4;       �	� �Yc�A�*

lossWd�>KNb       �	N}�Yc�A�*

loss���>MKk       �	��Yc�A�*

loss@��><z�       �	ŭ�Yc�A�*

loss���>Q̿       �	kb�Yc�A�*

lossm?T�       �	��Yc�A�*

lossnT�>\�&�       �	#��Yc�A�*

lossH�R>.U}�       �	&4�Yc�A�*

lossA!>��HZ       �	���Yc�A�*

lossS'(?d�b�       �	�k�Yc�A�*

loss���>mU�       �	 ��Yc�A�*

loss��>H_�       �	��Yc�A�*

loss�w�>�T8�       �	�%�Yc�A�*

loss�^�>�� �       �	���Yc�A�*

loss��>�Y#=       �	�W	�Yc�A�*

loss�d�>�j�       �	<�	�Yc�A�*

loss�$n>׿�T       �	Ő
�Yc�A�*

loss��=>�o�7       �	)�Yc�A�*

lossf�>
�i�       �	��Yc�A�*

loss��>��E       �	�g�Yc�A�*

loss]��>t���       �	��Yc�A�*

loss{1�>���       �	L��Yc�A�*

loss�G�>W9�       �	VI�Yc�A�*

loss�ԑ>��\�       �	y��Yc�A�*

loss�" ?�� �       �	]��Yc�A�*

loss_B�>��l�       �	�)�Yc�A�*

loss��A><P�b       �	���Yc�A�*

loss�mT>�ْ       �	��Yc�A�*

loss��>�oCd       �	��Yc�A�*

losso�?s��}       �	K��Yc�A�*

lossqϙ>Ѹ�       �	��Yc�A�*

lossv�>bg�       �	$&�Yc�A�*

losso��>�9�       �	*��Yc�A�*

loss�F[>9��[       �	�d�Yc�A�*

loss{s>�l)�       �	>�Yc�A�*

lossF̖>p��       �	��Yc�A�*

loss��>_$gv       �	]S�Yc�A�*

loss���>�H�U       �	7��Yc�A�*

lossb�>D�Q(       �	ҧ�Yc�A�*

loss��>��       �	[�Yc�A�*

lossJ3K>���;       �	y�Yc�A�*

loss��X>�H��       �	ۢ�Yc�A�*

loss�>��"�       �	�G�Yc�A�*

loss�o�>��C�       �	6��Yc�A�*

loss�7�>�<&�       �	~�Yc�A�*

lossū4>�Xz�       �	��Yc�A�*

loss(��>�-�       �	���Yc�A�*

loss���>e��(       �	P�Yc�A�*

loss7u>���       �	q�Yc�A�*

loss��?�n"       �	��Yc�A�*

loss,D>K0       �	,E �Yc�A�*

loss��>�^I�       �	� �Yc�A�*

loss�'�>Z�ގ       �	��!�Yc�A�*

lossI��>����       �	�8"�Yc�A�*

loss]|�>ڄ��       �	��"�Yc�A�*

loss�P?�IN�       �	��#�Yc�A�*

loss=�>Wj�2       �	y#$�Yc�A�*

loss��->0l��       �	��$�Yc�A�*

loss�Ӛ>n<b�       �	�g%�Yc�A�*

loss  �>4uj       �	�&�Yc�A�*

loss��>����       �	7�&�Yc�A�*

loss:�>��       �	�B'�Yc�A�*

lossz2�>�q��       �	��'�Yc�A�*

loss�X�>H0qa       �	.�(�Yc�A�*

loss=HO>"�8       �	�1)�Yc�A�*

lossT��>I�Sw       �	 �)�Yc�A�*

lossFג>跪h       �	9*�Yc�A�*

loss�~V>���       �	�"+�Yc�A�*

loss<�M>[B[       �	F�+�Yc�A�*

lossH9>�u��       �	;o,�Yc�A�*

loss8�X>�~�       �	|-�Yc�A�*

loss���>h�       �	D�-�Yc�A�*

loss��>f$ȅ       �	K.�Yc�A�*

loss&�O>tR        �	e�.�Yc�A�*

loss�r+>�ߊy       �	��/�Yc�A�*

loss�UY>T&ߛ       �	�'0�Yc�A�*

lossϤ�=����       �	��0�Yc�A�*

loss�x�=]���       �	!w1�Yc�A�*

loss�~>D~$�       �	2�Yc�A�*

loss�/>���       �	��2�Yc�A�*

lossQ��=l�C       �	�3�Yc�A�*

lossO*_>���       �	�+4�Yc�A�*

loss�9F>��       �	N�4�Yc�A�*

loss��>KZ�A       �	��5�Yc�A�*

lossEO�>]�0�       �	�Y6�Yc�A�*

loss�:�>m�/       �	��7�Yc�A�*

lossx�>J�N       �	�_8�Yc�A�*

loss
Y�>����       �	�9�Yc�A�*

loss�D>a3R       �	¤9�Yc�A�*

loss��U>j��a       �	�<:�Yc�A�*

loss�	�>�HI       �	T�<�Yc�A�*

loss�->�َ"       �	A=�Yc�A�*

loss/�s>���       �	��=�Yc�A�*

loss� >�b��       �	�q>�Yc�A�*

lossc�>��3       �	�
?�Yc�A�*

loss��p>ݱ^       �	��?�Yc�A�*

lossS5D>o��       �	q@�Yc�A�*

lossAĔ>��	       �	|
A�Yc�A�*

lossj�>n�8�       �	n�A�Yc�A�*

loss�+F>x���       �	�:B�Yc�A�*

loss��D>N�       �	��B�Yc�A�*

loss��B>L���       �	jhC�Yc�A�*

loss���>d55\       �	}D�Yc�A�*

lossd�R>
Ԥ/       �	J�D�Yc�A�*

loss���=p��       �	��E�Yc�A�*

loss��>���       �	sF�Yc�A�*

loss�2�>RO%       �	wG�Yc�A�*

lossȨ&>��T�       �	
�G�Yc�A�*

lossCX�>{       �	�[H�Yc�A�*

lossM�E>)��'       �	��H�Yc�A�*

loss㣄>j�.       �	ԝI�Yc�A�*

loss�Q�>q���       �	�SJ�Yc�A�*

loss6E?^՛�       �	4�K�Yc�A�*

loss:[�>��2       �	I�L�Yc�A�*

loss�/>�}�       �	�<M�Yc�A�*

loss>dy�       �	vN�Yc�A�*

lossO�7>���P       �	��O�Yc�A�*

loss�B>�9�"       �	hP�Yc�A�*

loss��:>u�6       �	1Q�Yc�A�*

loss��:>H�g       �	��Q�Yc�A�*

loss�~x>G��,       �	MR�Yc�A�*

lossp��>B���       �	�R�Yc�A�*

lossQ�>v��o       �	׆S�Yc�A�*

lossf�>Sϱ�       �	iT�Yc�A�*

loss1��>�sɞ       �	ǼT�Yc�A�*

loss
qG>�r�       �	�]U�Yc�A�*

lossd�>�A0b       �	��U�Yc�A�*

loss�:�>�޻       �	J�V�Yc�A�*

lossf%�=��-�       �	��W�Yc�A�*

loss�(>�t�v       �	IMX�Yc�A�*

losst�=>�K�L       �	.�X�Yc�A�*

loss�!?Ę�       �	�xY�Yc�A�*

loss���>:�L       �	Z�Yc�A�*

lossf�J>�w"	       �	
�Z�Yc�A�*

loss���>���       �	QJ[�Yc�A�*

lossq�z>��_       �	{�[�Yc�A�*

loss���>˸٘       �	
�\�Yc�A�*

loss��^>��       �	�B]�Yc�A�*

loss�u>> G�       �	�]�Yc�A�*

loss�v�=<Yä       �	(�^�Yc�A�*

loss�T>�-�       �	L_�Yc�A�*

loss�2>�]Ȋ       �	��_�Yc�A�*

loss�WE>�ڟ�       �	 R`�Yc�A�*

loss�S>�@�       �	d�`�Yc�A�*

lossJ�>*�O&       �	��a�Yc�A�*

loss�=M>2��       �	�(b�Yc�A�*

loss|J>�>'       �	��b�Yc�A�*

loss�Ë>g���       �	�Zc�Yc�A�*

loss*0B>ۈx�       �	'�c�Yc�A�*

loss_�=�G�       �	��d�Yc�A�*

lossX7�>����       �	>e�Yc�A�*

loss$[�>Z���       �	��e�Yc�A�*

losscڙ>S��       �	Qkf�Yc�A�*

loss4-�>���       �	g�Yc�A�*

loss�%&>���       �	��g�Yc�A�*

loss}.�>���       �	�Fh�Yc�A�*

loss�P>��_g       �	J�h�Yc�A�*

lossT�[>���k       �	ςi�Yc�A�*

lossڇA>^��       �	�&j�Yc�A�*

loss`��>q��       �	�j�Yc�A�*

loss�1I>��Ui       �	Uk�Yc�A�*

loss��>��l�       �	|�k�Yc�A�*

loss���>���h       �	&pm�Yc�A�*

loss vx>��X�       �	Wn�Yc�A�*

loss�F>�1�       �	��n�Yc�A�*

loss|(>jJ3�       �	=Fo�Yc�A�*

loss��>���o       �	)%p�Yc�A�*

lossw��>a��       �	��p�Yc�A�*

lossl��>P`��       �	aq�Yc�A�*

loss�4>��!       �	�&r�Yc�A�*

loss'">��Ca       �	%�r�Yc�A�*

lossW��=�[�       �	ms�Yc�A�*

lossO��=Z��       �	�t�Yc�A�*

loss��G>\%�9       �	�t�Yc�A�*

loss>:>#!�?       �	.Yu�Yc�A�*

loss�lW>Ɲ8�       �	ҋv�Yc�A�*

loss��>-p?       �	�$w�Yc�A�*

losstCa>��;4       �	#�w�Yc�A�*

loss��=�       �	[x�Yc�A�*

loss�{�>5�C       �	Z�x�Yc�A�*

loss`+�>1��H       �	��y�Yc�A�*

lossA'�>��=R       �	h%z�Yc�A�*

lossē�>�,*a       �	D�z�Yc�A�*

loss���=fĲ�       �	�a{�Yc�A�*

loss�)>�k�       �	D�{�Yc�A�*

lossM{�>D�c�       �	K�|�Yc�A�*

loss3ۈ>5��       �	�(}�Yc�A�*

loss��>-!�       �	l�}�Yc�A�*

loss�"�>�{ޛ       �	�n~�Yc�A�*

loss�e3>]��D       �	z�Yc�A�*

loss�58>q�+�       �	���Yc�A�*

loss���=�{&�       �	(`��Yc�A�*

loss�
�>�\       �	���Yc�A�*

loss�b>�3F       �	q���Yc�A�*

loss3�z>z���       �	�G��Yc�A�*

loss(1>�0&#       �	�܂�Yc�A�*

loss'�>��f"       �	�s��Yc�A�*

loss�	t>�ֳ�       �	���Yc�A�*

lossR�>pK��       �	֨��Yc�A�*

loss/$�>5+L1       �	P��Yc�A�*

loss�U�>�� 5       �	���Yc�A�*

loss�ӝ>�:�       �	���Yc�A�*

lossjё>�y�       �	UN��Yc�A�*

loss�y�=|�V       �	�퇿Yc�A�*

loss�&�=�B;�       �	ݖ��Yc�A�*

loss�I>>n���       �	�4��Yc�A�*

loss��2>����       �	�ɉ�Yc�A�*

loss�>%@S)       �	�i��Yc�A�*

loss�|G>�q7       �	l%��Yc�A�*

loss��8>pK&       �	�ċ�Yc�A�*

loss�z�>��B       �	� ��Yc�A�*

loss��>Պ�z       �	^���Yc�A�*

lossQ�v>�W�       �	�I��Yc�A�*

lossCX$>,�ު       �	��Yc�A�*

lossAC>�n,�       �	����Yc�A�*

loss�
>��`�       �	�%��Yc�A�*

loss���=��F
       �	�ϐ�Yc�A�*

loss���>��A�       �	{j��Yc�A�*

lossq�g>J��N       �	� ��Yc�A�*

loss�)">~ױ�       �	���Yc�A�*

loss��>����       �	9B��Yc�A�*

lossE�|>6H��       �	����Yc�A�*

lossŵ�=�FVX       �	����Yc�A�*

lossd��=�cE�       �	!"��Yc�A�*

lossY=>���       �	۾��Yc�A�*

loss#̟=�X�>       �	�`��Yc�A�*

loss��>���       �	���Yc�A�*

loss,b�>�F       �	����Yc�A�*

lossü�>$߽       �	�D��Yc�A�*

loss��>eG{�       �	 �Yc�A�*

loss��`>Gh�       �	����Yc�A�*

loss�u>�v��       �	�"��Yc�A�*

loss��>���/       �	"�Yc�A�*

loss�t?V�bP       �	�X��Yc�A�*

loss-�>��m       �	�Yc�A�*

loss�[�=Ig�       �	M���Yc�A�*

losss��>�PvY       �	?��Yc�A�*

loss�ET>����       �	����Yc�A�*

loss=<.>�Б       �	�S��Yc�A�*

lossH��=�hx       �	�ힿYc�A�*

loss�lE>3���       �	쇟�Yc�A�*

loss��>��}       �	%��Yc�A�*

loss!n�>��̱       �	c���Yc�A�*

loss*��>s�`�       �	R��Yc�A�*

loss�BB>�9�       �	G衿Yc�A�*

lossD-�>"�       �	����Yc�A�*

lossL�d>����       �	H��Yc�A�*

loss���=�P�       �	ɬ��Yc�A�*

loss��=�f�/       �	y>��Yc�A�*

loss��>�w��       �	�֤�Yc�A�*

loss��A>{��       �	R}��Yc�A�*

loss	v�=��D       �	��Yc�A�*

lossҌ�=��Q�       �	���Yc�A�*

loss�̯=�Uu       �	�A��Yc�A�*

loss� +>6:�       �	�٧�Yc�A�*

lossX->d�?�       �	\u��Yc�A�*

loss3�=�"�       �	�+��Yc�A�*

lossEV>+��       �	Vة�Yc�A�*

loss���>PoB       �	Lq��Yc�A�*

loss���>trj       �	���Yc�A�*

lossv=n>��;       �	ʫ�Yc�A�*

loss��>��M^       �	�j��Yc�A�*

loss�X=p��       �	J���Yc�A�*

loss�H�=�wv�       �	iV��Yc�A�*

lossH:\>����       �	���Yc�A�*

loss�C>��O�       �	���Yc�A�*

loss6�f>�曡       �	�H��Yc�A�*

loss��>��NW       �	#��Yc�A�*

lossA��=LWz�       �	����Yc�A�*

loss $E>'�+�       �	�^��Yc�A�*

loss�HU>ʹ�       �	e ��Yc�A�*

loss܄�=/�<f       �	0���Yc�A�*

loss�q	>���@       �	VD��Yc�A�*

loss�U�>�D�+       �	B봿Yc�A�*

loss$�:>z.x/       �	����Yc�A�*

lossqF�=	za�       �	K;��Yc�A�*

loss��V>[i�|       �	dX��Yc�A�*

loss>�EƳ       �	U���Yc�A�*

loss֦&>�>��       �	秸�Yc�A�*

loss��4>���       �	�U��Yc�A�*

loss�`�=~���       �	�Yc�A�*

loss�M�=�e��       �	A���Yc�A�*

loss�~�=����       �	�B��Yc�A�*

loss� F>�>�       �	�ܻ�Yc�A�*

loss�έ=��p       �	�ϼ�Yc�A�*

loss�+>#vO       �	�i��Yc�A�*

loss/V>|G�       �	��Yc�A�*

lossؑ�>+?,T       �	�,��Yc�A�*

loss-=�=#��b       �	�ȿ�Yc�A�*

loss7�&>y!E�       �	�c��Yc�A�*

loss��>���       �	���Yc�A�*

loss�<>3�ϡ       �	c���Yc�A�*

loss�b�=ݠ�       �	C7¿Yc�A�*

lossQ%E>��%       �	��¿Yc�A�*

loss�"�=��N       �	tÿYc�A�*

loss��g>�:�E       �	rĿYc�A�*

loss1�$>�C�#       �	��ĿYc�A�*

loss�s�=*��L       �	�UſYc�A�*

loss�c&>�##�       �	�ſYc�A�*

loss��=���       �		�ƿYc�A�*

lossR+I>���       �	�TǿYc�A�*

loss&OD>� x       �	e�ǿYc�A�*

loss�S_>��V�       �	\�ȿYc�A�*

loss���=�\�G       �	�IɿYc�A�*

loss��>)�M       �	M�ɿYc�A�*

lossH��=�ۗ       �	a�ʿYc�A�*

loss-�>��V       �	'M˿Yc�A�*

loss�0�=�:        �	��˿Yc�A�*

loss�F�=A�:�       �	#�̿Yc�A�*

lossϛ`=xg�       �	�<ͿYc�A�*

loss�J�=,��       �	:οYc�A�*

losst�>F�L|       �	��οYc�A�*

lossH� >�۝�       �	��ϿYc�A�*

lossږ�=����       �	n1пYc�A�*

lossH�n>����       �	��пYc�A�*

lossC"Z>7U��       �	B{ѿYc�A�*

lossdY�=�b       �	�ҿYc�A�*

loss�V�=�~p       �	'�ҿYc�A�*

loss��K>����       �	IcӿYc�A�*

loss��=��       �	hԿYc�A�*

loss���<@E�q       �	��ԿYc�A�*

loss���=@�[�       �	�LտYc�A�*

lossV�=�       �	�kֿYc�A�*

loss.V=���(       �	\׿Yc�A�*

loss��B=�dA       �	�׿Yc�A�*

loss�B2=�3�       �	�5ؿYc�A�*

lossՑ�>5��C       �	�ؿYc�A�*

lossݤ={�       �	ZgٿYc�A�*

loss�t�<K,�~       �	��ٿYc�A�*

losss{p=6��       �	��ڿYc�A�*

loss��3>^���       �	�4ۿYc�A�*

loss��Z>��i�       �	`�ۿYc�A�*

loss�5>:���       �	�^ܿYc�A�*

lossV�=i�Y{       �	^�ܿYc�A�*

lossZ��=���       �	֋ݿYc�A�*

lossa?��       �	1(޿Yc�A�*

loss;շ<��vN       �	��޿Yc�A�*

loss;�=D�h�       �	�_߿Yc�A�*

loss��=4�4       �	H�߿Yc�A�*

lossça>m���       �	��Yc�A�*

loss�>���       �	�B�Yc�A�*

loss�>X=ۨ_�       �	z��Yc�A�*

loss�F>��u]       �	���Yc�A�*

loss>�}{�       �	 &�Yc�A�*

loss�27>S֗�       �	��Yc�A�*

losss7a>��8�       �	se�Yc�A�*

lossS�X>�n	6       �	��Yc�A�*

loss@=c>;�3       �	a��Yc�A�*

loss��>�Đ       �	�A�Yc�A�*

losss�>��6       �	��Yc�A�*

loss�y�>v|�^       �	�p�Yc�A�*

loss��1>0�Q�       �	��Yc�A�*

loss�"N=K���       �	d��Yc�A�*

loss�>!��c       �	8N�Yc�A�*

lossJ`>��HH       �	���Yc�A�*

lossT�	>���       �	���Yc�A�*

lossfЅ=��|�       �	&�Yc�A�*

loss��'>�^�       �	���Yc�A�*

loss��>�O"�       �	�P�Yc�A�*

loss�:�=3�pn       �	���Yc�A�*

loss�f�=\�kd       �	M���Yc�A�*

lossT6=s�l�       �	�x�Yc�A�*

loss-�>id6l       �	/�Yc�A�*

loss.(�=�둊       �	��Yc�A�*

loss��=!
r�       �	9B�Yc�A�*

losss�=
.��       �	���Yc�A�*

lossr{	>qI�       �	`w�Yc�A�*

loss��=L�~F       �	=�Yc�A�*

loss�q<>V��       �	���Yc�A�*

loss.�I=Ǟ�       �	T;�Yc�A�*

loss�v�=\O       �	���Yc�A�*

loss��=<mJ�       �	���Yc�A�*

lossｕ=@ �       �	-]��Yc�A�*

loss�'>"}��       �	���Yc�A�*

loss�>��       �	4���Yc�A�*

loss�d]>w	        �	Q2��Yc�A�*

lossjz2=�zR       �	I���Yc�A�*

loss�}s=k�h       �	���Yc�A�*

loss���=�<V�       �	�/��Yc�A�*

loss��e>\��       �	c���Yc�A�*

lossc�>��Ð       �	B{��Yc�A�*

lossi�=K��       �	.��Yc�A�*

lossG�=_�i�       �	���Yc�A�*

loss��=-lt�       �	\Z��Yc�A�*

loss/2 >��       �	����Yc�A�*

loss%A>�       �	k��Yc�A�*

loss�M�=���       �	� �Yc�A�*

lossa�M>�j.(       �	���Yc�A�*

loss��>9���       �	Ig�Yc�A�*

loss���>@b�       �	���Yc�A�*

loss��=���       �	Ė�Yc�A�*

lossϼA>�d�       �	Z/�Yc�A�*

loss��=��M       �	���Yc�A�*

loss�>�~D       �	�^ �Yc�A�*

loss��>s=ZC       �	�� �Yc�A�*

loss��/>eF��       �	y�!�Yc�A�*

loss��>��I�       �	{/"�Yc�A�*

loss��f=�ML�       �	��"�Yc�A�*

loss�7�=�6��       �	al#�Yc�A�*

loss�>o�Q�       �	�$�Yc�A�*

loss�!>��M       �	��$�Yc�A�*

loss�!�=n�|       �	�=%�Yc�A�*

loss/^>�:�+       �	V�%�Yc�A�*

loss(��=D�7�       �	Ve&�Yc�A�*

lossv�.>(*/^       �	&�&�Yc�A�*

loss���=���       �	{�'�Yc�A�*

loss�D�>k�,H       �	�o(�Yc�A�*

loss�*J>
�5t       �	G)�Yc�A�*

loss҃�>_�oM       �	k�)�Yc�A�*

lossC8�=)�`�       �	�;*�Yc�A�*

loss�5> ?;       �	N�*�Yc�A�*

loss�6>�L��       �	�k+�Yc�A�*

loss�9>�H�       �	P,�Yc�A�*

loss�O�=��#       �	O�,�Yc�A�*

lossng�=�5��       �	�h-�Yc�A�*

lossQ>>7T       �	8.�Yc�A�*

loss���=$��       �	K�.�Yc�A�*

loss<.>\�D       �	��/�Yc�A�*

loss4��=�4$       �	[@0�Yc�A�*

lossSQ>۝6�       �	�0�Yc�A�*

loss\��>K�       �	�s1�Yc�A�*

loss���=?f4       �	�2�Yc�A�*

loss>�>v$�\       �	S�2�Yc�A�*

lossF\=�#�       �	�E3�Yc�A�*

lossL�i>��b�       �	@�3�Yc�A�*

lossţ>�
I       �	�4�Yc�A�*

loss�v�>6��.       �	w/5�Yc�A�*

loss}bX>��z       �	2�5�Yc�A�*

lossa.5>��~       �	��6�Yc�A�*

loss.� >Tw��       �	zn7�Yc�A�*

loss�c�>+��       �	%8�Yc�A�*

loss�V>b̤�       �	�k9�Yc�A�*

loss�z>��       �	�:�Yc�A�*

loss�)>f5�       �	�;�Yc�A�*

loss#>5>g��       �	�;�Yc�A�*

loss��=/�       �	q=�Yc�A�*

loss��=�xh�       �	��=�Yc�A�*

lossC��=Sz'D       �	jk>�Yc�A�*

lossz�0>���       �	�?�Yc�A�*

loss;A>�L�"       �	��?�Yc�A�*

lossW\?|0�       �	�H@�Yc�A�*

loss��>�ѵ�       �	��@�Yc�A�*

loss!�4=:�t       �	.�A�Yc�A�*

losst��=�td�       �	�8B�Yc�A�*

loss�J�=(p�       �	��B�Yc�A�*

loss4�L>]�o       �	C�C�Yc�A�*

lossj�<>����       �	%=D�Yc�A�*

loss�>(���       �	7�D�Yc�A�*

loss�T�=`E
M       �	|E�Yc�A�*

loss�F�=��>       �	�%F�Yc�A�*

loss�>����       �	u�F�Yc�A�*

loss��>|{v       �	vG�Yc�A�*

lossR�=���       �	�H�Yc�A�*

loss�\}>K~�       �	��H�Yc�A�*

loss3�>��Z�       �	�ZI�Yc�A�*

loss'ׁ>o ��       �	7�I�Yc�A�*

loss��>@��       �	ʣJ�Yc�A�*

loss�>��t�       �	F?K�Yc�A�*

lossA�=rh'       �	��K�Yc�A�*

lossZ��=/i�3       �	��L�Yc�A�*

loss�*>��)�       �	�8M�Yc�A�*

loss���=�N��       �	 _N�Yc�A�*

loss5�>f�q       �	<�N�Yc�A�*

loss��>��k       �	��O�Yc�A�*

lossJ3�>�9*�       �	&mP�Yc�A�*

loss��=*�#�       �	�+Q�Yc�A�*

loss���=wÞ�       �	��Q�Yc�A�*

loss*c,>�b       �	�bR�Yc�A�*

loss�C>Sw�       �	iQS�Yc�A�*

loss��a>c�       �	/0T�Yc�A�*

loss��=ZI�       �	A�T�Yc�A�*

lossn��=nQ�       �	χU�Yc�A�*

lossF�>����       �	�#V�Yc�A�*

loss��>�8~�       �	��V�Yc�A�*

lossX�4>F<�       �	�tW�Yc�A�*

lossŧ�=�S��       �	�X�Yc�A�*

loss��=�>�       �	�X�Yc�A�*

lossv\h>�H�       �	}[Y�Yc�A�*

loss�e>θ��       �	��Y�Yc�A�*

loss�>I�R�       �	��Z�Yc�A�*

loss�V�>�<v�       �	YM[�Yc�A�*

loss<��>�<P�       �	��[�Yc�A�*

lossH��=D�o       �	��\�Yc�A�*

lossl>��VR       �	�R]�Yc�A�*

lossc�>J�U�       �	D�]�Yc�A�*

loss�>�>��       �	��^�Yc�A�*

loss�:>l�3B       �	�D_�Yc�A�*

loss�Ά=�:�8       �	@O`�Yc�A�*

loss3Ɍ=9��       �	>a�Yc�A�*

loss�9�>��l       �	�a�Yc�A�*

loss$P>y�K       �	Kb�Yc�A�*

loss|�F>
~>�       �	��b�Yc�A�*

loss��1>�衻       �	��c�Yc�A�*

loss�V�=o��       �	�'d�Yc�A�*

loss#��=]�V�       �	~�d�Yc�A�*

loss?�t>��Q       �	fe�Yc�A�*

lossJ>��:(       �	f�Yc�A�*

loss(Z|>!o�       �	��f�Yc�A�*

loss�	>^p�       �	Zg�Yc�A�*

loss_�>���       �	
�g�Yc�A�*

loss��=!�j4       �	<�h�Yc�A�*

loss���='��       �	�8i�Yc�A�*

loss���=�AV       �	��i�Yc�A�*

loss�b,>�Y�       �	�j�Yc�A�*

loss�>�|G�       �	�+k�Yc�A�*

loss��=��u�       �	+�k�Yc�A�*

loss��=�gTV       �	�al�Yc�A�*

loss�ZA> B�       �	�m�Yc�A�*

loss��x>�IS�       �	Un�Yc�A�*

loss~��>��       �	��n�Yc�A�*

loss���=Z}Q(       �	�fo�Yc�A�*

loss��]==��       �	5p�Yc�A�*

loss��6>2I��       �	X�p�Yc�A�*

loss�RZ>=��       �	8Lq�Yc�A�*

lossՕ>TD��       �	q�q�Yc�A�*

lossZ�]>�A�       �	b�r�Yc�A�*

loss�4^>��͛       �	�)s�Yc�A�*

loss}�>gZ`6       �	�s�Yc�A�*

lossb� >�C       �	t�Yc�A�*

loss�h=MK�       �	�$u�Yc�A�*

loss�k=�/       �	�dv�Yc�A�*

loss��!>��B7       �	S�w�Yc�A�*

lossa�'>1��       �	]Rx�Yc�A�*

loss(�">A��       �	��x�Yc�A�*

loss�~�=��8V       �	��y�Yc�A�*

loss]�=�L       �	3�z�Yc�A�*

loss�=����       �	�j{�Yc�A�*

loss�N�<�=-�       �	�"|�Yc�A�*

loss3�>U��       �	�}�Yc�A�*

loss�8>��#�       �	�}�Yc�A�*

loss��=;�       �	��~�Yc�A�*

loss��n>�S�G       �	'O�Yc�A�*

lossD�=~��i       �	T ��Yc�A�*

lossRN�=U�v       �	Y���Yc�A�*

loss]n9=r���       �	�U��Yc�A�*

lossT��=��#t       �	t��Yc�A�*

loss;}�=�L��       �	���Yc�A�*

loss<�g=�8<�       �	*7��Yc�A�*

loss��>-i��       �	����Yc�A�*

loss�2W>�PE       �	 ��Yc�A�*

loss�� >>�R       �	���Yc�A�*

lossm�
>�\�       �	Ĵ��Yc�A�*

loss���=�m�m       �	�d��Yc�A�*

loss���=\r��       �	����Yc�A�*

loss>'>�Ki       �	ᴇ�Yc�A�*

loss�~�>�6{3       �	�\��Yc�A�*

loss��=���h       �	b��Yc�A�*

loss۬V>�5V       �	���Yc�A�*

loss=q�=�рE       �	1E��Yc�A�*

lossR�2>�$�=       �	�܊�Yc�A�*

loss��F=h YG       �	Y���Yc�A�*

loss�>o�[       �	�A��Yc�A�*

loss��=�!}�       �	Y݌�Yc�A�*

loss��=�k�       �	0���Yc�A�*

loss��>,]       �	a��Yc�A�*

loss�P�=���       �	汎�Yc�A�*

loss��=ƺ3$       �	[a��Yc�A�*

lossR��=j��       �	V(��Yc�A�*

lossJ��=�       �	ː�Yc�A�*

loss3(�=�5h�       �	oe��Yc�A�*

lossݐ >t�       �	q��Yc�A�*

losszԲ<i��       �	ʤ��Yc�A�*

loss�:>�i�       �	NB��Yc�A�*

loss��V>@��       �	�ܓ�Yc�A�*

lossdZ>ry(       �	v��Yc�A�*

loss�)�=v{�0       �	{��Yc�A�*

loss`v\=G���       �	����Yc�A�*

loss��$=n?"|       �	�D��Yc�A�*

loss_%*>��8�       �	���Yc�A�*

lossuT�=��C       �	����Yc�A�*

loss//�=���       �	?S��Yc�A�*

loss� �=uʭ       �	��Yc�A�*

lossJV=]��1       �	f���Yc�A�*

loss@��=�ѓ�       �	�Z��Yc�A�*

loss�??>���F       �	����Yc�A�*

loss8>8���       �	~���Yc�A�*

lossM��=�f�        �	�,��Yc�A�*

lossD<'>	�ψ       �	{��Yc�A�*

lossMd>�g�       �	����Yc�A�*

lossz !>�O.O       �	B��Yc�A�*

loss3��=�       �	�ݞ�Yc�A�*

loss� �=)~�D       �	dx��Yc�A�*

loss��>uEr�       �	��Yc�A�*

lossj�>\��%       �	����Yc�A�*

loss�G�=��|�       �	Z��Yc�A�*

lossS��=0��[       �	Q���Yc�A�*

lossE�=Z�Hg       �	����Yc�A�*

loss7� >��`@       �	�3��Yc�A�*

lossX�d=��       �	<ۣ�Yc�A�*

loss@Ѩ=vbZ       �	ٙ��Yc�A�*

loss��=�t2�       �	D3��Yc�A�*

loss!�=��       �	���Yc�A�*

loss���=ƅ�       �	V~��Yc�A�*

loss��=,���       �	���Yc�A�*

loss��|=q       �	����Yc�A�*

loss���=Kjt�       �	H��Yc�A�*

loss��=Zo��       �	]ߨ�Yc�A�*

loss��=L�Џ       �	Gu��Yc�A�*

loss�Y�=#4VC       �	M��Yc�A�*

loss��W=sxL�       �	���Yc�A�*

loss��`=ao��       �	�H��Yc�A�*

loss1>�8l       �	r��Yc�A�*

loss��=6谮       �	�w��Yc�A�*

loss;Z�=A���       �	o��Yc�A�*

lossdG>r��R       �	/���Yc�A�*

loss�=�>���       �	�9��Yc�A�*

loss��>�{�x       �	�Ϯ�Yc�A�*

loss��<2M�       �	t��Yc�A�*

loss)�@>�0       �	�	��Yc�A�*

loss��o>�!�       �	穰�Yc�A�*

loss� ->C��       �	�?��Yc�A�*

loss��=��}       �	���Yc�A�*

loss� q=�:�       �	����Yc�A�*

loss��>v��       �	jm��Yc�A�*

loss�=���       �	R��Yc�A�*

loss#=�=�p��       �	ګ��Yc�A�*

losseq�=��V�       �	E��Yc�A�*

loss6�T=[��H       �	���Yc�A�*

loss���=͞�4       �	߶�Yc�A�*

loss�*>�C�       �	�ط�Yc�A�*

loss��=;z*�       �	���Yc�A�*

loss4>�ǥI       �	����Yc�A�*

losso>QS�       �	Q���Yc�A�*

loss`A>�=�[       �	�J��Yc�A�*

loss��->$B�J       �	���Yc�A�*

loss��>�2��       �	�/��Yc�A�*

loss;��=^|D4       �	�
��Yc�A�*

lossJ=�=��       �	!ɾ�Yc�A�*

loss{:>��'        �	͐��Yc�A�*

loss&0>���       �	I���Yc�A�*

loss���=��{       �	Sw��Yc�A�*

lossZ8>�R�P       �	� ��Yc�A�*

loss�>Ss3"       �	)	��Yc�A�*

lossS�=��"n       �	a���Yc�A�*

loss�C�=�@�       �	ظ��Yc�A�*

loss��=�>�_       �	�W��Yc�A�*

loss��0>��e�       �	���Yc�A�*

loss:�7>�$�E       �	
���Yc�A�*

loss@w5>!���       �	cA��Yc�A�*

loss��B>���       �	����Yc�A�*

loss�b=�`��       �	Ow��Yc�A�*

loss�i=�GJ       �	���Yc�A�*

loss�>%��       �	.���Yc�A�*

loss5 >c�y       �	�^��Yc�A�*

loss�$>��*       �	����Yc�A�*

loss�	�=�=��       �	���Yc�A�*

loss�g�=( [       �	����Yc�A�*

loss��=�8̭       �	Ad��Yc�A�*

loss���=%��       �	����Yc�A�*

loss�ǡ>��CF       �	}���Yc�A�*

loss�a�=$Y       �	9^��Yc�A�*

loss b>>�g�       �	����Yc�A�*

lossD%>��9�       �	d���Yc�A�*

loss�CT>u��v       �	�)��Yc�A�*

loss/�&>C��       �	����Yc�A�*

loss;�K>���p       �	����Yc�A�*

loss�5*>+3 �       �	�u��Yc�A�*

lossx��=�P9       �	���Yc�A�*

lossA�=r��       �	>���Yc�A�*

loss�ю=���       �	)���Yc�A�*

loss4N>'F`t       �	�3��Yc�A�*

loss.]%>Ti��       �	E���Yc�A�*

loss �e>��Ju       �	�y��Yc�A�*

lossxy�=��*       �	� ��Yc�A�*

loss��D=�{�       �	����Yc�A�*

loss��5>�~       �	sJ��Yc�A�*

loss���=)�:	       �	?���Yc�A�*

loss�H�=��r�       �	�}��Yc�A�*

lossNY�=�O8       �	k��Yc�A�*

loss.�>S-�       �	n���Yc�A�*

lossW��=S͠       �	G:��Yc�A�*

loss3� >�.�\       �	����Yc�A�*

loss\��= S�[       �	zo��Yc�A�*

loss�`�=����       �	���Yc�A�*

loss�� >��ge       �	G���Yc�A�*

loss>�ʆ�       �	\��Yc�A�*

loss�RB=��F       �	0���Yc�A�*

loss��"=&���       �	����Yc�A�*

loss�f�=���       �	���Yc�A�*

loss���=���4       �	,��Yc�A�*

loss�=:=����       �		���Yc�A�*

loss+>�Gؿ       �	�V��Yc�A�*

loss�C>]u�<       �	����Yc�A�*

loss�o�=i��M       �	���Yc�A�*

loss�>Z�M[       �	���Yc�A�*

loss���=9xo       �	����Yc�A�*

loss�T>�˽       �	�>��Yc�A�*

loss���=h�h�       �	B���Yc�A�*

lossl}�=�Ĕt       �	�S��Yc�A�*

loss
�>x�4�       �	���Yc�A�*

loss�OC>w�       �	���Yc�A�*

loss��=����       �	�J��Yc�A�*

loss��>H�hC       �	���Yc�A�*

lossj�=�  J       �	K���Yc�A�*

loss_�&>^��X       �	h^��Yc�A�*

lossus�=�V4�       �	����Yc�A�*

loss �=��-�       �	���Yc�A�*

loss_ �=j �F       �	T9��Yc�A�*

loss��8>��=�       �	���Yc�A�*

lossi>uG?*       �	j��Yc�A�*

loss�Q<=�-j       �	W��Yc�A�*

lossV��=�+��       �	���Yc�A�*

lossU&>�Ʉ�       �	V��Yc�A�*

loss�!>:��       �	�K��Yc�A�*

loss��U=����       �	��Yc�A�*

lossX|2>���       �	����Yc�A�*

loss�>��F       �	1]��Yc�A�*

loss��~=g\D       �	����Yc�A�*

loss�>1�,@       �	ϣ��Yc�A�*

loss��>����       �	�<��Yc�A�*

loss�(>.���       �	���Yc�A�*

loss�JA>4O       �	"��Yc�A�*

loss�B�>��i�       �	����Yc�A�*

loss�"�=�.��       �	�n��Yc�A�*

loss3�c=����       �	���Yc�A�*

lossj.'>uU��       �	����Yc�A�*

loss��>+wC�       �	u<��Yc�A�*

loss�=�P-       �	����Yc�A�*

loss
|�=θ��       �	���Yc�A�*

loss��=i7;a       �	�$��Yc�A�*

loss6=	�+X       �	w���Yc�A�*

loss׃�=me�j       �	����Yc�A�*

loss1yK=��1       �	j1��Yc�A�*

loss��>A�@       �	����Yc�A�*

loss���=�	��       �	~q �Yc�A�*

loss�a=S�#�       �	�	�Yc�A�*

loss���=\%H       �	6��Yc�A�*

loss�q�=2&��       �	�L�Yc�A�*

lossrӘ=n��       �	��Yc�A�*

loss�>�l       �	���Yc�A�*

loss ��=b8iR       �	�(�Yc�A�*

loss��\>綦b       �	���Yc�A�*

lossF4'>�<a       �	[_�Yc�A�*

loss���=�w�       �	
��Yc�A�*

loss�Y">�vë       �	���Yc�A�*

loss3�=M�ԙ       �	-�Yc�A�*

lossJP)>�>�       �	��Yc�A�*

lossscp=�7�*       �	�n�Yc�A�*

lossv��=��P       �	W
	�Yc�A�*

lossF��=۷�B       �	�	�Yc�A�*

loss��=�e�       �	�S
�Yc�A�*

loss$�>���       �	��
�Yc�A�*

loss�X=�;<�       �	��Yc�A�*

loss��P=�3�       �	�7�Yc�A�*

lossp�=��(       �	�D�Yc�A�*

loss�%�=�eqE       �	���Yc�A�*

lossE�L=���       �	��Yc�A�*

lossJf�=��b       �	��Yc�A�*

loss�@�=����       �	��Yc�A�*

lossSй=p��       �	L�Yc�A�*

loss�-2>�8O�       �	��Yc�A�*

loss�>e��       �	��Yc�A�*

lossԡ>4��{       �	��Yc�A�*

loss!|E>x���       �	���Yc�A�*

lossM��=�6'H       �	5B�Yc�A�*

loss��=-+       �	A�Yc�A�*

loss��m=��2�       �	���Yc�A�*

loss���=�C�g       �	�I�Yc�A�*

loss���=�n       �	���Yc�A�*

loss�<>CVY       �	|��Yc�A�*

loss8��=ݸJ�       �	*�Yc�A�*

loss�{)>.       �	E��Yc�A�*

loss�i>��_�       �	x�Yc�A�*

loss��=Z�C       �	�Yc�A�*

loss.�=���       �	ۤ�Yc�A�*

loss��=�fjE       �	�;�Yc�A�*

loss��5>��y&       �	W�Yc�A�*

loss�>g1g       �	w��Yc�A�*

loss�N>��       �	9�Yc�A�*

loss�`=����       �	A��Yc�A�*

lossz��=e���       �	+k�Yc�A�*

loss�ZL=�3J�       �	�7�Yc�A�*

lossz*;=�A<�       �	p��Yc�A�*

loss >K�'�       �	�x�Yc�A�*

loss̛�=�lmH       �	6" �Yc�A�*

loss��=˯V�       �	� �Yc�A�*

loss��>3$5�       �	qs!�Yc�A�*

lossS�>�V�       �	�"�Yc�A�*

loss�gw=�I��       �	�"�Yc�A�*

loss�r&>����       �	�P#�Yc�A�*

loss�(>:�
D       �	t�#�Yc�A�*

loss*U�=l#�{       �	L�$�Yc�A�*

loss\�6>&�<�       �	�+%�Yc�A�*

loss
�=g}��       �	��%�Yc�A�*

loss�o�=� f       �	X�&�Yc�A�*

lossq�=��       �	�1'�Yc�A�*

loss�>�{T�       �	)�'�Yc�A�*

loss�,>��J�       �	am(�Yc�A�*

loss��>c��       �	l	)�Yc�A�*

loss��=��*       �	1�)�Yc�A�*

loss.�>w�A�       �	LO*�Yc�A�*

loss]�=��k"       �	o�*�Yc�A�*

lossd�=)K{�       �	�+�Yc�A�*

loss�>o���       �	 ),�Yc�A�*

loss-��=^S��       �	B�,�Yc�A�*

loss;>��       �	�d-�Yc�A�*

lossf/	>䔵�       �	`:.�Yc�A�*

lossS>��       �	��.�Yc�A�*

loss�.H>c�	�       �	�r/�Yc�A�*

loss�3�=��Qg       �	�
0�Yc�A�*

lossx�l>q�S       �	��0�Yc�A�*

loss�K>�       �	hB1�Yc�A�*

loss�)B>�ј�       �	��1�Yc�A�*

loss�V�<���       �	�2�Yc�A�*

loss!��=`E�       �	R&3�Yc�A�*

loss_��=
�t       �	��3�Yc�A�*

loss[�=��       �	;o4�Yc�A�*

loss�^�=�?=       �	Z5�Yc�A�*

loss��=�K>�       �	N�5�Yc�A�*

loss���=xA��       �	�t6�Yc�A�*

loss[�c>��       �	�u7�Yc�A�*

loss�6>YRq       �	K8�Yc�A�*

loss�1>kk�       �	�{9�Yc�A�*

loss@l�=��       �	\�:�Yc�A�*

loss`1>Ȳ��       �	χ;�Yc�A�*

loss���=ᎋ�       �	��<�Yc�A�*

loss��%>��w�       �	��=�Yc�A�*

loss��L>���       �	�>�Yc�A�*

loss(0>��sp       �	a�?�Yc�A�*

loss�
�=�b�5       �	v�@�Yc�A�*

loss[��=S:%�       �	��A�Yc�A�*

loss��=���       �	�bB�Yc�A�*

lossd�=e��       �	5C�Yc�A�*

loss�W�=t�}       �	��C�Yc�A�*

lossƔP=R���       �	ȱD�Yc�A�*

lossa�]=):�       �	��E�Yc�A�*

loss.�=R��'       �	�\F�Yc�A�*

loss���>�?��       �	2tG�Yc�A�*

loss��>R��_       �	�XH�Yc�A�*

loss�=�M       �	�hI�Yc�A�*

loss�zS=t��       �	�,J�Yc�A�*

lossχ�=V��       �	��J�Yc�A�*

loss��>4�&�       �	zK�Yc�A�*

loss`�>���A       �	�L�Yc�A�*

loss\>����       �	`�L�Yc�A�*

loss%u=*��       �	�DM�Yc�A�*

loss�H9>T8��       �	3�M�Yc�A�*

loss��(>��^f       �	�sN�Yc�A�*

loss��Y=0-��       �	�O�Yc�A�*

loss�;�=r<�       �	G�O�Yc�A�*

lossM�!>|Գ       �	�VP�Yc�A�*

loss}/O>T*L�       �	s�P�Yc�A�*

loss3�C>=��       �	)�Q�Yc�A�*

lossd�>Ϸ�       �	+5R�Yc�A�*

loss���=ߞ�       �	�R�Yc�A�*

loss3��=�dno       �	+iS�Yc�A�*

loss,/�=�\       �	�T�Yc�A�*

loss95�=m!�       �	�T�Yc�A�*

loss�M	=�Ǐ�       �	�MU�Yc�A�*

lossA`=��Q       �	�U�Yc�A�*

loss\P>L55;       �	�}V�Yc�A�*

loss�=1$�       �	+W�Yc�A�*

loss�Z=��.�       �	�W�Yc�A�*

loss��=���       �	�~X�Yc�A�*

loss���=��       �	�Y�Yc�A�*

loss�t>(�"       �	�Y�Yc�A�*

loss�	= ��       �	RGZ�Yc�A�*

loss�=�N       �	��Z�Yc�A�*

loss�D�=6�Q�       �	�[�Yc�A�*

loss�>O%       �	�9\�Yc�A�*

loss >.�       �	�\�Yc�A�*

loss��n=Q�|U       �	�k]�Yc�A�*

loss�S=��       �	�^�Yc�A�*

loss:��<Ԧ�       �	�^�Yc�A�*

loss�|>Y�N�       �	�g_�Yc�A�*

loss�w�=��       �	�`�Yc�A�*

loss�d>�*�<       �	��`�Yc�A�*

lossh�>/���       �	%\a�Yc�A�*

loss\��=Cc~-       �	j�a�Yc�A�*

loss�>_��       �	��b�Yc�A�*

lossB�=�U�       �	�9c�Yc�A�*

lossk >�ы�       �	d�Yc�A�*

loss���=^Y%       �	i�d�Yc�A�*

loss�>l� �       �	0Ge�Yc�A�*

loss� �=�M4P       �	v�e�Yc�A�*

loss���=�jy       �	�|f�Yc�A�*

loss�u'>wܖ�       �	 g�Yc�A�*

lossO�S>0ְ�       �	��g�Yc�A�*

lossX��=l�       �	�Sh�Yc�A�*

loss�}�=~�       �	�h�Yc�A�*

lossJ"�=�ߐ       �	��i�Yc�A�*

lossj1p=D`�       �	�>j�Yc�A�*

loss6M=�4i�       �	��j�Yc�A�*

loss��>�Y�       �	�k�Yc�A�*

loss�cs=�B]�       �	vl�Yc�A�*

loss���=���       �	J�l�Yc�A�*

lossA�">��V       �	�Mm�Yc�A�*

loss��=�Gw0       �	��m�Yc�A�*

loss�rH=��a       �	��n�Yc�A�*

loss�bu=�'��       �	�>o�Yc�A�*

loss�M�=W�9       �	��o�Yc�A�*

loss��K>=�g�       �	~rp�Yc�A�*

loss�/_=��-�       �	�
q�Yc�A�*

loss��>c^�>       �	�q�Yc�A�*

lossr��=�b*�       �	_Er�Yc�A�*

loss@A�=�"S�       �	L�r�Yc�A�*

lossܻ>��0       �	1xs�Yc�A�*

loss���=�'�0       �	t�Yc�A�*

loss���=���       �	�t�Yc�A�*

loss%��<��y>       �	NGu�Yc�A�*

loss�b2>g��0       �	��u�Yc�A�*

loss�Q�=���       �	~�v�Yc�A�*

lossm�!>*fa0       �	/w�Yc�A�*

loss���=��m^       �	[�w�Yc�A�*

lossp=4�Z.       �	.ux�Yc�A�*

loss�Cq=Q��"       �	]�y�Yc�A�*

loss���=(P4$       �	6z�Yc�A�*

loss?j*=�2�S       �	��z�Yc�A�*

loss� >����       �	��{�Yc�A�*

lossâ�<#N̂       �	��|�Yc�A�*

loss#�=ħ��       �	�7}�Yc�A�*

loss���=�r       �	��}�Yc�A�*

loss6y�=�&b       �	E�~�Yc�A�*

loss4�=��Æ       �	n�Yc�A�*

loss�>��m�       �	.��Yc�A�*

loss�m>~�H�       �	�@��Yc�A�*

loss�NF=�e0�       �	9��Yc�A�*

loss.�=K�       �	����Yc�A�*

loss���=� �>       �	c��Yc�A�*

loss@��<A��       �	#���Yc�A�*

loss6�=�w�P       �	����Yc�A�*

loss�׍=�'��       �	'.��Yc�A�*

loss��=�3��       �	Ä�Yc�A�*

loss|i�<Q9�       �	[��Yc�A�*

lossd�v=��6�       �	����Yc�A�*

loss�΍<bP��       �	���Yc�A�*

loss!��=���@       �	�B��Yc�A�*

lossO,=A�7�       �	&��Yc�A�*

loss4�g<���        �	�y��Yc�A�*

loss:G4<X>.       �	�8��Yc�A�*

loss8��=(/�       �	͉�Yc�A�*

lossS�>w�       �	�c��Yc�A�*

lossn�g=�:�7       �	����Yc�A�*

loss��<sr!�       �	����Yc�A�*

lossE�g=srk       �	80��Yc�A�*

lossO?�ߌ       �	�Č�Yc�A�*

loss�ƌ=�*}        �	z��Yc�A�*

loss6�=�A_�       �	R*��Yc�A�*

lossJ��=nn       �	#���Yc�A�	*

lossz�1>��[       �	���Yc�A�	*

loss���==��       �	߈��Yc�A�	*

loss�}}=U��       �	y��Yc�A�	*

loss4=>���       �	����Yc�A�	*

loss�>�2t       �	�K��Yc�A�	*

loss��>�f}�       �	���Yc�A�	*

losso�>����       �	'���Yc�A�	*

lossz| >�#�8       �	w��Yc�A�	*

losst�>�5 �       �	k��Yc�A�	*

lossf�>��:�       �	����Yc�A�	*

loss��>���       �	�f��Yc�A�	*

loss1��=[V�H       �	���Yc�A�	*

loss��B>)B+�       �	Ƨ��Yc�A�	*

loss��=�W\n       �	�=��Yc�A�	*

loss�">��|       �	�ט�Yc�A�	*

loss(�=ϩ�h       �	.s��Yc�A�	*

loss��=�I
%       �	���Yc�A�	*

loss��u=Iq       �	峚�Yc�A�	*

lossDG�=�;�n       �	+N��Yc�A�	*

loss,d=8f9�       �	���Yc�A�	*

loss��(=���R       �	Q���Yc�A�	*

loss�1�=p;�
       �	6 ��Yc�A�	*

loss�=�d\       �	и��Yc�A�	*

losss�=���       �	�U��Yc�A�	*

lossA��=s�B�       �	���Yc�A�	*

loss���=Fw�       �	䅟�Yc�A�	*

loss[�=��M       �	��Yc�A�	*

loss#E=f�lu       �	@���Yc�A�	*

loss:_�=��K       �	+k��Yc�A�	*

lossΨ1>�h��       �	V��Yc�A�	*

loss3:�<r$��       �	⮢�Yc�A�	*

loss2�=Ș�4       �	T��Yc�A�	*

loss͐�=����       �	�3��Yc�A�	*

lossoc3=[7�q       �	Iդ�Yc�A�	*

loss���=�IFP       �	=|��Yc�A�	*

loss��=:%       �	���Yc�A�	*

lossT�=�Ol.       �	����Yc�A�	*

loss4�~=H�p       �	+P��Yc�A�	*

loss�R@=�z       �	���Yc�A�	*

loss���=�P��       �	����Yc�A�	*

loss�dD>K��&       �	���Yc�A�	*

loss֍>+�OV       �	(���Yc�A�	*

loss���=,��[       �	~U��Yc�A�	*

loss�?�=<s2       �	|��Yc�A�	*

loss
M=��+O       �	����Yc�A�	*

loss@��=�(       �	r1��Yc�A�	*

loss���=X櫣       �	�ά�Yc�A�	*

lossq�=� f       �	�d��Yc�A�	*

loss�o>���       �	ۋ��Yc�A�	*

loss@N�=��B*       �	*��Yc�A�	*

losshE>�9�D       �	K���Yc�A�	*

lossc>9�U�       �	�f��Yc�A�	*

loss�H�=n�"       �	���Yc�A�	*

loss�(=�(6�       �	ߨ��Yc�A�	*

loss,��=����       �	�A��Yc�A�	*

loss8��=;Ą�       �	U���Yc�A�	*

lossZc�=�D��       �	�t��Yc�A�	*

loss�3>��D       �	o��Yc�A�	*

loss��<�c�\       �	���Yc�A�	*

loss��=���F       �	�@��Yc�A�	*

loss|w�=��y       �	����Yc�A�	*

loss��h>��       �	lv��Yc�A�	*

loss��=Q��       �	���Yc�A�	*

loss�_	>r��       �	@���Yc�A�	*

loss���<���       �	�=��Yc�A�	*

loss6�=,��t       �	o���Yc�A�	*

loss�!Q=Le�       �	/l��Yc�A�	*

loss�i>r��       �	���Yc�A�	*

loss&��={�
       �	ܡ��Yc�A�	*

loss�$�=o.I�       �	|H��Yc�A�	*

loss���=����       �	����Yc�A�	*

loss��E>���        �	}���Yc�A�	*

loss���=s���       �	#k��Yc�A�	*

loss):=��0b       �	���Yc�A�	*

loss�{�=*�o       �	Ǽ��Yc�A�	*

lossr=t��       �	�]��Yc�A�	*

loss��=F	�%       �	|���Yc�A�	*

loss��=�d       �	-]��Yc�A�	*

lossW�l=��vH       �	e��Yc�A�	*

loss��=�LŹ       �	���Yc�A�	*

loss�;>}��       �	2<��Yc�A�	*

loss�c�=W���       �	����Yc�A�	*

losss��<�� �       �	
���Yc�A�	*

loss:��=����       �	�&��Yc�A�	*

lossV��<�u�       �	����Yc�A�	*

loss��=,}d�       �	�]��Yc�A�	*

lossĲ�>vz(4       �	����Yc�A�	*

lossIo>��       �	o���Yc�A�	*

loss	 
>���       �	Ҏ��Yc�A�	*

loss��=x�#�       �	�"��Yc�A�	*

loss�lK=7C�       �	Y���Yc�A�	*

loss�	>C��o       �	Mg��Yc�A�	*

loss�{�=�=�       �	���Yc�A�	*

loss���=�,B+       �	����Yc�A�	*

loss>��<�       �	�~��Yc�A�	*

lossm��=�VWJ       �	���Yc�A�	*

loss��=���       �	���Yc�A�	*

loss�'�<��A       �	VD��Yc�A�	*

loss	h=Q���       �	���Yc�A�	*

lossRJ�=�N�q       �	ro��Yc�A�	*

loss!q�=H""       �	i��Yc�A�	*

loss;�>3aI�       �	���Yc�A�	*

loss�ɵ=� �       �	�J��Yc�A�	*

lossS�<�+�       �	;���Yc�A�	*

loss:��<�֚M       �	�y��Yc�A�	*

lossw�=����       �	���Yc�A�	*

loss���=C	h       �	����Yc�A�	*

loss�=�$�       �	�Q��Yc�A�	*

losseM�=� [�       �	���Yc�A�	*

loss�>��W       �	q���Yc�A�	*

loss�^P=1�       �	1&��Yc�A�	*

loss,�>uj�       �	����Yc�A�	*

loss�=���+       �	�]��Yc�A�	*

lossV�=_d'�       �	���Yc�A�	*

loss3�>l�       �	;���Yc�A�	*

loss\6�=qK       �	6��Yc�A�	*

loss*�N>��n       �	
���Yc�A�	*

loss��;>��e       �	N���Yc�A�	*

lossc��=Mm*       �	.9��Yc�A�	*

loss�@,>�       �	#���Yc�A�	*

loss���=�io�       �	2r��Yc�A�	*

loss��=y�T0       �	���Yc�A�	*

loss�a�=aS�       �	���Yc�A�	*

lossM1�=��~�       �	e���Yc�A�	*

loss��=o[#h       �	�u��Yc�A�	*

loss��J> �۬       �	�M��Yc�A�	*

loss�P=gܦ       �	)��Yc�A�	*

losse�,=�.��       �	/���Yc�A�
*

loss[�>�M�       �	�S��Yc�A�
*

loss�a$>�pi�       �	
M��Yc�A�
*

loss?��=v��       �	#��Yc�A�
*

loss��A=�D�       �	L���Yc�A�
*

lossxo�=�xp       �	To��Yc�A�
*

lossm�=D��;       �	L���Yc�A�
*

lossE�o=�N       �	�k �Yc�A�
*

loss�I�=���       �	���Yc�A�
*

loss}�=���       �	&T�Yc�A�
*

loss\1�=��;>       �	�O�Yc�A�
*

lossJ�>u=��       �	���Yc�A�
*

loss���=4؉�       �	���Yc�A�
*

lossߠ�=$R�       �	ۦ�Yc�A�
*

losst�>"�k       �	C�Yc�A�
*

loss(�$>J��:       �	���Yc�A�
*

loss.^A=:I�       �	��Yc�A�
*

loss8�h=��M       �	�0�Yc�A�
*

loss|u�=Q�       �	���Yc�A�
*

loss�>K>}~EF       �	z	�Yc�A�
*

losst�=vc��       �	�
�Yc�A�
*

lossԱ$=c�#�       �	�
�Yc�A�
*

losswu2=̽�       �	B]�Yc�A�
*

loss+1>��Ǜ       �	��Yc�A�
*

loss]t>=~9�       �	���Yc�A�
*

loss���=�C�d       �	�S�Yc�A�
*

loss�l�=����       �	��Yc�A�
*

loss�{9=ϟ��       �	H��Yc�A�
*

loss�}�=j3p�       �	�"�Yc�A�
*

loss��,>&f�       �	���Yc�A�
*

loss�{�= �@       �	'f�Yc�A�
*

loss�E>����       �	.�Yc�A�
*

loss��=HK�,       �	���Yc�A�
*

lossI��=����       �	nR�Yc�A�
*

lossا-=���       �	���Yc�A�
*

loss,��=�Kyc       �	Ŏ�Yc�A�
*

loss�Β=w���       �	�&�Yc�A�
*

loss��=�H�       �	���Yc�A�
*

loss��=�mV	       �	C��Yc�A�
*

loss�]�=����       �	(�Yc�A�
*

lossT�=g��       �	��Yc�A�
*

loss8��=��       �	�i�Yc�A�
*

loss�Ș=�m��       �	`�Yc�A�
*

lossj�#>�9�       �	���Yc�A�
*

loss��"=����       �	,E�Yc�A�
*

lossa�;=�3�+       �	��Yc�A�
*

loss�z>�M�3       �	
��Yc�A�
*

loss�>�o�       �	B^�Yc�A�
*

loss�u&=^Dh�       �	���Yc�A�
*

lossHA>��ޭ       �	p��Yc�A�
*

loss�P�=�C��       �	�3�Yc�A�
*

loss���=cfK`       �	>#�Yc�A�
*

lossR��=���S       �	G��Yc�A�
*

loss#��=OWT=       �	Ic�Yc�A�
*

loss+�=t�
       �	? �Yc�A�
*

loss��>���&       �	� �Yc�A�
*

loss#2=��N(       �	��!�Yc�A�
*

lossF�a=b7ݭ       �	eS"�Yc�A�
*

loss��4=N;%&       �	8�"�Yc�A�
*

loss^��=��       �	A�#�Yc�A�
*

loss�%�=kqF�       �	�`$�Yc�A�
*

loss��-<��&�       �	�$�Yc�A�
*

losss�p=_2u       �	%�%�Yc�A�
*

loss�>�El       �	�&�Yc�A�
*

loss\��=R�^�       �	>'�Yc�A�
*

loss�� >�}s�       �	U�'�Yc�A�
*

loss��]=�	Y�       �	��(�Yc�A�
*

loss��=�a6       �	�&)�Yc�A�
*

loss���<���       �	`�)�Yc�A�
*

loss�:,=�"7       �	Uh*�Yc�A�
*

loss-��=�W'�       �	Z+�Yc�A�
*

loss��g=���       �	�+�Yc�A�
*

lossW�)>dA'       �	X,�Yc�A�
*

loss�"�=ߡv       �	x�,�Yc�A�
*

loss��=gr+       �	]�-�Yc�A�
*

loss=�=��       �	�%.�Yc�A�
*

loss1Y�<N�X�       �	_�.�Yc�A�
*

loss���=O)�        �	}z/�Yc�A�
*

loss.�	>���s       �	70�Yc�A�
*

loss8>�?u�       �	��0�Yc�A�
*

loss���=��z�       �	�^1�Yc�A�
*

loss�f�=����       �	d2�Yc�A�
*

loss\�=�F ]       �	5�2�Yc�A�
*

loss1��=3h�k       �	[3�Yc�A�
*

losse�'=w�       �	�3�Yc�A�
*

loss-�=�4�}       �	h�4�Yc�A�
*

lossZ�}=z�{�       �	�;5�Yc�A�
*

lossC=(4f       �	z�5�Yc�A�
*

lossQ��=�g       �	56�Yc�A�
*

loss%x�=f�       �	�7�Yc�A�
*

loss�*�=��       �	m�7�Yc�A�
*

loss�ݹ=`��       �	��8�Yc�A�
*

lossḒ=/h^�       �	�:9�Yc�A�
*

loss�:>�f�       �	{-:�Yc�A�
*

loss)�=�^�'       �	W[;�Yc�A�
*

loss(=��k�       �	��;�Yc�A�
*

loss��=�o       �	5�<�Yc�A�
*

lossp<>V�H�       �	JF=�Yc�A�
*

loss}��=��0       �	N�=�Yc�A�
*

loss�=���g       �	ˠ>�Yc�A�
*

lossݯ<�lz�       �	�O?�Yc�A�
*

loss_�<p
�       �	��?�Yc�A�
*

loss�N�=?�W       �	p�@�Yc�A�
*

lossҙo=>?�`       �	�<A�Yc�A�
*

loss(�=�6�       �	��A�Yc�A�
*

loss���<Z��       �	��B�Yc�A�
*

loss�G=,Bv       �	�zC�Yc�A�
*

loss�&P=~��&       �	Z.D�Yc�A�
*

loss���=l_	       �	C�D�Yc�A�
*

loss*(�=�\�.       �	p{E�Yc�A�
*

lossA�>K���       �	1'F�Yc�A�
*

loss}>����       �	�F�Yc�A�
*

lossn��=���       �	x`G�Yc�A�
*

loss2`�=��p       �	&:H�Yc�A�
*

loss��=@�9�       �	��H�Yc�A�
*

loss.�]<<2O       �	��I�Yc�A�
*

lossF�z=
��       �	�3J�Yc�A�
*

lossWD>��RG       �	�K�Yc�A�
*

loss#p0>�u�       �	��K�Yc�A�
*

lossW��=���       �	3OL�Yc�A�
*

loss��=�j��       �	�L�Yc�A�
*

loss_��=��Rg       �	��M�Yc�A�
*

loss<�~=9N\�       �	5N�Yc�A�
*

lossԕ�=�s,�       �	��N�Yc�A�
*

loss�=�P�       �	�jO�Yc�A�
*

loss?=vV�s       �	-P�Yc�A�
*

loss��Q=�p�L       �	ǡP�Yc�A�
*

lossʲ=�?z�       �	�GQ�Yc�A�
*

lossO�=��2       �	��Q�Yc�A�*

loss�d�=c��H       �	2�R�Yc�A�*

lossdD�=Toq�       �	�9S�Yc�A�*

loss~�=Tu��       �	��S�Yc�A�*

lossDA�=�M l       �	��T�Yc�A�*

lossS19=�iw       �	%"U�Yc�A�*

loss�~�=�2�+       �	��U�Yc�A�*

lossh�>+OΊ       �	�\V�Yc�A�*

loss(-�=�q�d       �	"�V�Yc�A�*

loss�ۯ=����       �	p�W�Yc�A�*

loss� <>5 �       �	9X�Yc�A�*

loss�>����       �	��X�Yc�A�*

loss�ot>�g{>       �	)wY�Yc�A�*

loss�	�<��e       �	� Z�Yc�A�*

loss�X>sq�       �	��Z�Yc�A�*

loss6��=�Br       �	?T[�Yc�A�*

loss��S>�o�#       �	E�[�Yc�A�*

lossf��=\�7�       �	�\�Yc�A�*

loss��=� �W       �	�W]�Yc�A�*

loss	��=u�Up       �	�^�Yc�A�*

loss���=���       �	f�^�Yc�A�*

lossю�=�_       �	�<_�Yc�A�*

loss��=n?�       �	�_�Yc�A�*

loss�I�=3�D�       �	@�`�Yc�A�*

lossl�<$8��       �	R,a�Yc�A�*

loss�)>��\       �	��a�Yc�A�*

loss�Q�=O��       �	�jb�Yc�A�*

loss��=��       �	!c�Yc�A�*

loss�ׇ=��38       �	�c�Yc�A�*

loss/�=̬j�       �	Gd�Yc�A�*

losshd�=�D�*       �	y�d�Yc�A�*

lossFX�=��=�       �	ߊe�Yc�A�*

loss�:=�e�       �	@0f�Yc�A�*

loss�E�=ҹPQ       �	O�f�Yc�A�*

loss�̽=^M�E       �	��g�Yc�A�*

loss�`�=���       �	�h�Yc�A�*

loss�T=�E��       �	h�h�Yc�A�*

losss�=�~�       �	/ji�Yc�A�*

loss
��=na��       �	j�Yc�A�*

lossK>̵�       �	�j�Yc�A�*

loss8N=*�K�       �	�Ek�Yc�A�*

loss��=�s�       �	��k�Yc�A�*

lossӊ�=Ciw{       �	��l�Yc�A�*

loss1=$>]y�       �	�hm�Yc�A�*

loss��->4�ֶ       �	*n�Yc�A�*

loss�%�=��       �	�n�Yc�A�*

loss�&�<k�       �	=o�Yc�A�*

loss���<b�ì       �	k�o�Yc�A�*

loss�7=���3       �	iop�Yc�A�*

loss�J�=��k�       �	Gq�Yc�A�*

loss4�>�ZO       �	��q�Yc�A�*

lossnzT=moE       �	}r�Yc�A�*

loss*��=�d��       �	Hs�Yc�A�*

loss#��=�U9z       �	�s�Yc�A�*

loss%�>v�=x       �	8It�Yc�A�*

loss<�-> p��       �	��t�Yc�A�*

lossf�=+ê-       �	�u�Yc�A�*

loss,�%>:ޘk       �	/jv�Yc�A�*

losss��<�!#�       �	Aw�Yc�A�*

lossqu>�Y��       �	��w�Yc�A�*

loss�U�=c�;       �	�Nx�Yc�A�*

loss�S>���2       �	)y�Yc�A�*

loss���=@�*�       �	&�y�Yc�A�*

loss ��=x�fa       �	��z�Yc�A�*

loss�=7Y��       �	�L{�Yc�A�*

loss��=�D       �	��|�Yc�A�*

loss��>㻒]       �	�7}�Yc�A�*

loss
�=���h       �	_�}�Yc�A�*

loss�D�>��;s       �	J{~�Yc�A�*

loss���= �       �	�#�Yc�A�*

lossd-7=�w>S       �	O[��Yc�A�*

loss4��=7}��       �	���Yc�A�*

losss9�=��zo       �	���Yc�A�*

loss�MV<x�Ԑ       �	���Yc�A�*

loss (=UTt�       �	<J��Yc�A�*

lossL�>u�       �	a߃�Yc�A�*

loss�/!=��}A       �	À��Yc�A�*

loss�WD>�Z�       �	O��Yc�A�*

loss�f�=���       �	b���Yc�A�*

loss!�x=
���       �	�W��Yc�A�*

loss��=��B       �	E���Yc�A�*

loss���=�ʐt       �	����Yc�A�*

loss�i<h�       �	�1��Yc�A�*

lossi'd<��#       �	p҈�Yc�A�*

loss���=XJ�8       �	�i��Yc�A�*

loss��=����       �	���Yc�A�*

loss���<�m�5       �	Y܊�Yc�A�*

losshl�=y�       �	�p��Yc�A�*

loss���={�       �	
��Yc�A�*

loss��V=xΧ       �	���Yc�A�*

loss3>C��       �	WC��Yc�A�*

lossػ>�>�       �	�ލ�Yc�A�*

loss|�>KT�       �	˅��Yc�A�*

loss���=ְ"�       �	d#��Yc�A�*

loss�P=��4�       �	v���Yc�A�*

loss�ޗ=Ծ�	       �	���Yc�A�*

lossŬ�=8G�       �	A��Yc�A�*

loss�=�v�5       �	�ޑ�Yc�A�*

loss��=
	��       �	�y��Yc�A�*

loss�<,<��J�       �	a��Yc�A�*

loss��=f�       �	����Yc�A�*

lossjs�=]_,       �	PT��Yc�A�*

lossT�=�R#]       �	���Yc�A�*

loss6O�=�3ir       �	7���Yc�A�*

lossq&>�/��       �	�:��Yc�A�*

lossM">=��g       �	�Җ�Yc�A�*

lossѶ�=���t       �	${��Yc�A�*

loss�=��ep       �	� ��Yc�A�*

lossx.�=��v       �	^���Yc�A�*

loss���=(p��       �	�b��Yc�A�*

loss\�<K{�:       �	)	��Yc�A�*

loss�%>[�0       �	����Yc�A�*

loss\�=��       �	�;��Yc�A�*

lossS�<X�       �	ћ�Yc�A�*

loss-��=�L]�       �	.s��Yc�A�*

loss��= Mh       �	���Yc�A�*

loss�_�=Z3       �	S���Yc�A�*

loss��>��`}       �	K��Yc�A�*

loss?7>U�       �	J��Yc�A�*

loss�A�=�2�       �	����Yc�A�*

lossS=�}�5       �	*:��Yc�A�*

loss3A�=�	�       �	�נ�Yc�A�*

lossd͒=C��$       �	�{��Yc�A�*

loss��r=��&W       �	]��Yc�A�*

loss�e�=iU�w       �	D1��Yc�A�*

lossa~�<�f�<       �	Qۣ�Yc�A�*

loss��=��=�       �	�u��Yc�A�*

loss/A�=a�
       �	J��Yc�A�*

lossA0�=�&�u       �	����Yc�A�*

loss��=�7�       �	$G��Yc�A�*

loss(>4�U,       �	��Yc�A�*

lossQ�S=>�r       �	���Yc�A�*

lossϷ< ���       �	�#��Yc�A�*

loss�t�=g@t�       �	����Yc�A�*

loss� �<�PW       �	�a��Yc�A�*

lossNW�=L�CR       �	X ��Yc�A�*

lossq-�<�x�       �	H���Yc�A�*

loss{}D>M�_�       �	%<��Yc�A�*

loss��=����       �	ܫ�Yc�A�*

loss�UZ=�k��       �	.���Yc�A�*

lossw �=��@R       �	�+��Yc�A�*

loss�W=�V�       �	Gƭ�Yc�A�*

loss�V>����       �	yZ��Yc�A�*

loss}r�<)w7�       �	R��Yc�A�*

loss��?=��VV       �	2���Yc�A�*

loss�8�=0|�*       �	*��Yc�A�*

loss���=sQ�A       �	�Ȱ�Yc�A�*

loss�?�=|�`       �	�c��Yc�A�*

loss]�=��       �	 ��Yc�A�*

lossA�=^C�       �	k���Yc�A�*

lossqH�=�m        �	�M��Yc�A�*

lossz7=/_�       �	(��Yc�A�*

loss[�A=G�n       �	͒��Yc�A�*

lossC�	>d�Z       �	f2��Yc�A�*

loss�|�=Hi       �	9ӵ�Yc�A�*

loss\��=��b�       �	?p��Yc�A�*

loss�)8>�lž       �	k��Yc�A�*

losst>Kї       �	���Yc�A�*

loss���=�{�l       �	�I��Yc�A�*

loss�>h��       �	2��Yc�A�*

loss�aV=���       �	o��Yc�A�*

lossA�l=���u       �	���Yc�A�*

loss�a:=5��       �	�C��Yc�A�*

loss:�=&��       �	O��Yc�A�*

lossҠ]=4s�       �	����Yc�A�*

loss�J�=�"��       �	r���Yc�A�*

loss�U=��       �	-?��Yc�A�*

loss�R�=?Qݳ       �	@۾�Yc�A�*

loss��>�l       �	,���Yc�A�*

loss���=B�c�       �	L��Yc�A�*

loss��-= o��       �	����Yc�A�*

loss3��=��       �	X��Yc�A�*

loss���=�m��       �	�*��Yc�A�*

lossP$>��|       �	2���Yc�A�*

lossN�=�~s2       �	`r��Yc�A�*

loss�S%=�?�       �	���Yc�A�*

lossF��=���       �	���Yc�A�*

loss�a=�V�       �	x���Yc�A�*

lossc>/=K�K�       �	HR��Yc�A�*

lossd��=�sq�       �	N���Yc�A�*

loss���=�S       �	���Yc�A�*

lossV>I=�"��       �	�8��Yc�A�*

loss;�=Tk�       �	����Yc�A�*

loss�-�=n�@x       �	����Yc�A�*

loss阳=��0       �	�&��Yc�A�*

loss.�=�z-       �	�(��Yc�A�*

loss�l=��&�       �	7���Yc�A�*

loss8f�=�s�       �	Y��Yc�A�*

loss�OU>	#$�       �	����Yc�A�*

loss�V=���G       �	ڌ��Yc�A�*

loss�"=�z.       �	�'��Yc�A�*

loss�+�=0���       �	����Yc�A�*

losss˸=b���       �	�b��Yc�A�*

loss&�o=���       �	����Yc�A�*

loss�=>!�3       �	>���Yc�A�*

loss#'�=㺞�       �	�W��Yc�A�*

lossK�==��t       �	���Yc�A�*

loss�!=I���       �	����Yc�A�*

loss؅�=��p       �	�$��Yc�A�*

loss�%�=��\�       �	e���Yc�A�*

loss�<>��       �	�`��Yc�A�*

loss(�=8���       �	�7��Yc�A�*

loss���=Y���       �	$���Yc�A�*

loss��=ʬ�D       �	�l��Yc�A�*

loss���=;/�       �	���Yc�A�*

loss���=���       �	̶��Yc�A�*

loss��>>����       �	�Y��Yc�A�*

loss��Z>�}�       �	���Yc�A�*

loss��>�#��       �	M���Yc�A�*

lossf��<f33�       �	�$��Yc�A�*

loss�=v��       �	���Yc�A�*

loss�ۗ=���2       �	\��Yc�A�*

loss�>=�m�       �	
���Yc�A�*

loss��p=ؖ��       �	>���Yc�A�*

loss��E=�5Q�       �	�9��Yc�A�*

loss*��=c��       �	,���Yc�A�*

loss�>��        �	��Yc�A�*

loss��o=Z^�       �	: ��Yc�A�*

loss�
>4`5       �	K���Yc�A�*

loss�L�=��#       �	�d��Yc�A�*

lossҀ�=�"       �	!��Yc�A�*

loss�3�=��}Y       �	^���Yc�A�*

loss �=���t       �	*8��Yc�A�*

losssO�=��1       �	f���Yc�A�*

loss!^�=(��       �	`w��Yc�A�*

loss�(R=�u       �	���Yc�A�*

loss�̮=� 	       �	����Yc�A�*

loss���=*�:!       �	�H��Yc�A�*

loss%�=��V�       �	e���Yc�A�*

loss���=��,7       �	�z��Yc�A�*

lossؔp=@���       �	"��Yc�A�*

lossz�8=�~5�       �	A���Yc�A�*

loss���<0�Y�       �	�V��Yc�A�*

loss��_>ou=�       �	����Yc�A�*

lossc>9>���       �	���Yc�A�*

loss:��=���       �	d;��Yc�A�*

lossV)�=Vf*8       �	n��Yc�A�*

loss��<�l�_       �	t���Yc�A�*

loss�T�=Ez*       �	O��Yc�A�*

loss wM>��       �	c���Yc�A�*

loss�C�=O3"�       �	 ���Yc�A�*

loss�1=1�       �	�5��Yc�A�*

lossN�>3q&       �	����Yc�A�*

loss��=�]�|       �	����Yc�A�*

loss=�s��       �	�F��Yc�A�*

losss=���       �	����Yc�A�*

lossi�=�6�b       �	A���Yc�A�*

lossm�]>Z��s       �	���Yc�A�*

losshq�=��S�       �	���Yc�A�*

loss�q>l�s       �	�^��Yc�A�*

loss�wM=";yF       �	����Yc�A�*

lossS��=S�?-       �	ŏ��Yc�A�*

loss6��=(��O       �	�S��Yc�A�*

loss_�a=d׿�       �	����Yc�A�*

loss�i:=T5�       �	ߌ��Yc�A�*

loss�m=K��       �	N(��Yc�A�*

lossl��=)n�U       �	����Yc�A�*

loss�R�=KeD       �	�a��Yc�A�*

loss�!�<�U�       �	�$��Yc�A�*

loss#�=�A.�       �	����Yc�A�*

loss�zN=�r��       �	ta��Yc�A�*

loss3�>�J��       �	���Yc�A�*

losslM:=��m�       �	9���Yc�A�*

loss=;�=<��	       �	pa��Yc�A�*

lossC�=��|       �	r��Yc�A�*

loss��=Q�g       �	���Yc�A�*

loss��J>�ix       �	bi��Yc�A�*

lossυ�=����       �	���Yc�A�*

loss��6<�Y1�       �	:���Yc�A�*

loss!�=PG��       �	�Z �Yc�A�*

loss.H�=�&�       �	)�Yc�A�*

loss8�\=���       �	¦�Yc�A�*

lossm�(>��z)       �	�L�Yc�A�*

lossġ�>:�)D       �	���Yc�A�*

loss��2=ĊR�       �	���Yc�A�*

lossƄ=Z1;       �	�/�Yc�A�*

loss��=�6l�       �	��Yc�A�*

loss)�=��z       �	�y�Yc�A�*

loss#�k=C7o�       �	�"�Yc�A�*

lossl�H=#���       �	S��Yc�A�*

loss���=�M�       �	�m�Yc�A�*

loss%E=�z��       �	�Yc�A�*

lossr�8=9�]�       �	7��Yc�A�*

loss��=���       �	K	�Yc�A�*

loss��=��Q       �	��	�Yc�A�*

loss���=Ҹ�D       �	�}
�Yc�A�*

loss��4=ZiZ       �	��Yc�A�*

lossh�;=�[��       �	!��Yc�A�*

loss��Q=�ShY       �	(H�Yc�A�*

loss,�>y�       �	R��Yc�A�*

loss���<� {I       �	ӆ�Yc�A�*

lossL=�*�,       �	d�Yc�A�*

loss���=E�IK       �	E��Yc�A�*

lossq�=4T�       �	}X�Yc�A�*

lossH\�<����       �	6�Yc�A�*

loss&4�<��l�       �	k��Yc�A�*

lossW�Y=��       �	�K�Yc�A�*

losset>�z}�       �	O�Yc�A�*

loss��X=�Ɍ       �	��Yc�A�*

loss7%�=�F �       �	��Yc�A�*

lossƸ=�=-g       �	R'�Yc�A�*

lossi��=���o       �	���Yc�A�*

lossO�=Q�64       �	�y�Yc�A�*

loss��=7�e       �	�(�Yc�A�*

loss��=h���       �	B��Yc�A�*

loss�� =\���       �	���Yc�A�*

lossj�n=�߉       �	o�Yc�A�*

loss���=<8Y       �	�Yc�A�*

loss�B�=�M��       �	3�Yc�A�*

lossi�=Q�SS       �	��Yc�A�*

loss���=�$       �	Ln�Yc�A�*

loss�/=CZ       �	%�Yc�A�*

loss?��={���       �	P��Yc�A�*

loss��=�(�@       �	�J�Yc�A�*

lossv�=_�BP       �	O��Yc�A�*

loss��<�0n�       �	�S�Yc�A�*

lossѬ�=�j��       �	���Yc�A�*

loss�b=����       �	�� �Yc�A�*

loss�ϕ=��=       �	�/!�Yc�A�*

lossVX=�(�       �	��!�Yc�A�*

loss��=�S�       �	=d"�Yc�A�*

loss�\�=/�`�       �	#�Yc�A�*

loss��=3U��       �	�#�Yc�A�*

loss$�Y=���       �	<J$�Yc�A�*

loss���=Ǥ4       �	��$�Yc�A�*

loss&�_<��:       �	��%�Yc�A�*

lossi�;��*       �	*&�Yc�A�*

loss3�<����       �	�&�Yc�A�*

loss�ז<�+�       �	UL'�Yc�A�*

loss���<�n�       �	�'�Yc�A�*

loss�l�=�,       �	7�(�Yc�A�*

loss�`�<�H       �	>!)�Yc�A�*

loss�s[=`P�p       �	�)�Yc�A�*

loss�e?<g�&�       �	rO*�Yc�A�*

loss1<2��       �	��*�Yc�A�*

lossA��;R�'       �	��+�Yc�A�*

lossa9\=��u       �	 (,�Yc�A�*

lossL��=�Z�D       �	6�,�Yc�A�*

loss:?�=>�%       �	ע-�Yc�A�*

loss���<�h�       �	!<.�Yc�A�*

loss]=���       �	�.�Yc�A�*

loss�1�>���z       �	�m/�Yc�A�*

loss*}9<��J�       �	�0�Yc�A�*

loss@�v=��n�       �	Z�0�Yc�A�*

loss[��=�/��       �	�:1�Yc�A�*

loss7(�=c>�;       �	��1�Yc�A�*

lossAC�=�1�       �	e�2�Yc�A�*

loss	�<m��       �	�E3�Yc�A�*

loss�d#>�v��       �	��3�Yc�A�*

loss�>�R�       �	��4�Yc�A�*

loss���<\C��       �	�65�Yc�A�*

loss�?>�l�       �	��5�Yc�A�*

loss�Nm=�{Xw       �	Nz6�Yc�A�*

lossJ>u�       �	Q7�Yc�A�*

loss���=źM�       �	��7�Yc�A�*

loss�">J�;       �	�b8�Yc�A�*

loss��=��       �	�9�Yc�A�*

loss�T8>�j�F       �	�9�Yc�A�*

lossQ��=��JM       �	��:�Yc�A�*

losse�\=����       �	{�;�Yc�A�*

lossxM�=�5�*       �	l=<�Yc�A�*

loss��`=f�.�       �	�=�Yc�A�*

loss���<nZ�       �	��=�Yc�A�*

loss�W�=S��       �	H�>�Yc�A�*

loss���=E��       �	�?�Yc�A�*

loss��<�r��       �	)@�Yc�A�*

loss f=� [       �	��@�Yc�A�*

loss�<�<��9I       �	�cA�Yc�A�*

losswrl=E��       �	c	B�Yc�A�*

loss�G=H�P�       �	ϢB�Yc�A�*

loss�r�=?�       �	�DC�Yc�A�*

lossm�=�^�       �	%�C�Yc�A�*

loss���<0�f       �	ǸD�Yc�A�*

loss�!�=%��       �	-ZE�Yc�A�*

loss�>ާ�$       �	��E�Yc�A�*

loss:�<'|��       �	6�F�Yc�A�*

loss4�=����       �	@2G�Yc�A�*

loss��=`�       �	J�G�Yc�A�*

lossMP�<�P��       �	:tH�Yc�A�*

loss#�=�P=�       �	<I�Yc�A�*

lossJE�=)�O       �	��I�Yc�A�*

loss=��=m�o�       �	�OJ�Yc�A�*

lossX-�<�=%n       �	��J�Yc�A�*

loss�=n)�i       �	h�K�Yc�A�*

loss�B=���       �	fL�Yc�A�*

lossm5>�t�V       �	�M�Yc�A�*

lossVc>=s��       �	��M�Yc�A�*

loss�S=���       �	DN�Yc�A�*

loss�<=��       �	�N�Yc�A�*

loss��S<i>p�       �	uvO�Yc�A�*

loss�D%=hP�       �	�P�Yc�A�*

lossS�7=� 1       �	.�P�Yc�A�*

loss"=��       �	��Q�Yc�A�*

lossd��=��B�       �	 �i�Yc�A�*

loss�Vu=z�?�       �	Z�j�Yc�A�*

lossN�>>K�@�       �	5�k�Yc�A�*

loss�>>�G�       �	�Hl�Yc�A�*

lossI�=��K       �	�4m�Yc�A�*

loss�6=���       �	�m�Yc�A�*

loss��<=Ё`�       �	zpn�Yc�A�*

loss�Je=��       �	�	o�Yc�A�*

loss��>۽�_       �	��o�Yc�A�*

lossѭ�=ls&       �	R`p�Yc�A�*

loss�@�<-�       �	�q�Yc�A�*

loss��=Y�q       �	��q�Yc�A�*

loss���=�Nss       �	�<r�Yc�A�*

loss��P>���       �	��r�Yc�A�*

loss�S�=\)��       �	�zs�Yc�A�*

loss�=��)�       �	�#t�Yc�A�*

loss3��;�-4H       �	o�t�Yc�A�*

loss s�=��^�       �	�u�Yc�A�*

loss@IB=[bbR       �	�"v�Yc�A�*

lossf8�=)l�       �	 �v�Yc�A�*

loss��=Ï�       �	gw�Yc�A�*

loss���=!<�       �	"x�Yc�A�*

loss��Q=��       �	K�x�Yc�A�*

loss���=��       �	�ry�Yc�A�*

loss1ݱ=zo��       �	�z�Yc�A�*

loss�`y=��       �	��z�Yc�A�*

loss�5�=��s�       �	Ϊ{�Yc�A�*

loss�h=7���       �	P�|�Yc�A�*

loss1�=��m       �	8}�Yc�A�*

loss�+�=���       �	�	~�Yc�A�*

lossjH�=̤�       �	��~�Yc�A�*

loss�ъ=�י       �	�s�Yc�A�*

lossͧ=䬄       �	��Yc�A�*

lossj!�=Ͽ��       �	����Yc�A�*

lossZ8=�	R       �	I���Yc�A�*

lossA!J=Ă��       �	�>��Yc�A�*

loss=r�{       �	X9��Yc�A�*

loss�1>�n&�       �	�܃�Yc�A�*

loss�e$>;�       �	Sz��Yc�A�*

loss�>�%O�       �	���Yc�A�*

loss�o�=ʇG       �	/���Yc�A�*

loss���=�U��       �	�c��Yc�A�*

loss���<�{�       �	^��Yc�A�*

lossoWk=Y�P        �	ȳ��Yc�A�*

loss��=���8       �	�Y��Yc�A�*

loss��f=��       �	���Yc�A�*

loss�6�=����       �	����Yc�A�*

lossKo=��p�       �	[��Yc�A�*

loss\<o=���+       �	|���Yc�A�*

lossmg�<�;��       �	?��Yc�A�*

loss;�C=:O^       �	&��Yc�A�*

loss(��=��%�       �	3���Yc�A�*

lossZ�o=$R68       �	���Yc�A�*

loss���>g6��       �	˺��Yc�A�*

lossNB=���       �	�^��Yc�A�*

lossvl<�O_�       �	���Yc�A�*

loss�{�;��       �	㩐�Yc�A�*

loss�=��n�       �	�>��Yc�A�*

loss�|~=�O%O       �	�ב�Yc�A�*

loss�M�=��$�       �	%���Yc�A�*

loss�j�=�P+i       �	�I��Yc�A�*

loss���=�       �	����Yc�A�*

loss}�=A���       �	��Yc�A�*

loss�=�=�̶       �	Ƈ��Yc�A�*

lossDg=��l'       �	�8��Yc�A�*

loss&8y=n���       �	fI��Yc�A�*

loss8d�=�W��       �	6��Yc�A�*

loss��W=��       �	����Yc�A�*

loss�U >d��       �	d��Yc�A�*

lossD�>�ݞe       �	����Yc�A�*

loss�8=��X	       �	o���Yc�A�*

loss��=C�'       �	G<��Yc�A�*

loss�u=�	d�       �	��Yc�A�*

loss:��=Fz       �	Dݜ�Yc�A�*

loss���=�!'�       �	����Yc�A�*

loss�x'=ǂ�       �	�5��Yc�A�*

loss���=�ho       �	�О�Yc�A�*

loss�I>��       �	�i��Yc�A�*

loss�b={S]�       �	tE��Yc�A�*

loss᳛=�	��       �	�ݠ�Yc�A�*

loss��+=N���       �	Tr��Yc�A�*

loss��=����       �	���Yc�A�*

lossF��=ς�M       �	����Yc�A�*

loss9#=�ν�       �	&S��Yc�A�*

loss5b�=�6X�       �	���Yc�A�*

loss�U�=zм4       �	tҤ�Yc�A�*

losso�b=�t       �	����Yc�A�*

loss�V�=z�1�       �	3��Yc�A�*

loss&p�=��&j       �	�Ц�Yc�A�*

loss�8Q=�J�       �	�r��Yc�A�*

lossfh�=���       �	���Yc�A�*

lossꤪ=��F�       �	V���Yc�A�*

loss5�=G�6�       �	�Z��Yc�A�*

loss�֔=(;\.       �	���Yc�A�*

loss�)�=E[       �	����Yc�A�*

loss�!=�Bt�       �	+5��Yc�A�*

loss�Gs=�F\       �	�Ϋ�Yc�A�*

loss�=	�2/       �	+k��Yc�A�*

losss�->pit       �	�	��Yc�A�*

loss�j�=��t       �	�ͭ�Yc�A�*

loss�V=d�n�       �	�q��Yc�A�*

loss�ݼ=��       �	S ��Yc�A�*

loss�_�=�1f       �	�̯�Yc�A�*

loss*o=Hg�?       �	�q��Yc�A�*

lossx4�=&�V=       �	���Yc�A�*

lossJJ�=7�       �	ú��Yc�A�*

lossT%=� �       �	%]��Yc�A�*

loss�$=��dW       �	���Yc�A�*

loss�'>�R       �	���Yc�A�*

loss�>��e�       �	<2��Yc�A�*

loss�g>�vG�       �	���Yc�A�*

loss\r|=���       �	���Yc�A�*

loss��_=��       �	W=��Yc�A�*

loss�=F��G       �	��Yc�A�*

loss�^=�l�       �	����Yc�A�*

loss%��=�I�       �	C=��Yc�A�*

loss|�]=�N�$       �	�ظ�Yc�A�*

loss>9�=4�۟       �	r��Yc�A�*

lossSC>{��       �	��Yc�A�*

loss�(�<E^o       �	.���Yc�A�*

loss& �=^�A�       �	uX��Yc�A�*

lossS�=I~��       �	G���Yc�A�*

loss�I>���       �	�2��Yc�A�*

lossڊ@=�c��       �	Խ�Yc�A�*

loss��<�~@       �	���Yc�A�*

loss�I=L�ŷ       �	�8��Yc�A�*

lossXL�=�Tr�       �	�տ�Yc�A�*

loss�G=K�
       �	���Yc�A�*

loss� �=Kݖ�       �	�^��Yc�A�*

lossv�=vԓ�       �	)��Yc�A�*

loss��=�
�       �	����Yc�A�*

loss��=.�OC       �	i���Yc�A�*

loss�I=���       �	�?��Yc�A�*

lossƨ@=��D       �	����Yc�A�*

lossc��=MM&y       �	ݗ��Yc�A�*

loss��u=�Գ�       �	QN��Yc�A�*

losswݺ=pj	�       �	����Yc�A�*

loss�Li="a       �	����Yc�A�*

loss? m=w�l       �	]Q��Yc�A�*

loss;�=	,       �	P���Yc�A�*

lossv ]<�5z       �	���Yc�A�*

lossz8=���       �	A��Yc�A�*

lossɥE=��L&       �	z���Yc�A�*

loss�y=��/~       �	����Yc�A�*

loss�c�==u��       �	QK��Yc�A�*

loss4��=3$�       �	6���Yc�A�*

lossݼ�=$X7       �	#���Yc�A�*

lossҵ�<�_��       �	q��Yc�A�*

lossD�*=P��#       �	���Yc�A�*

loss)�A=�X�H       �	�Q��Yc�A�*

loss7PT=կ��       �	5���Yc�A�*

loss\��=>       �	e���Yc�A�*

loss��=�vɣ       �	F(��Yc�A�*

loss��=��3�       �	r���Yc�A�*

loss��=���       �	���Yc�A�*

lossV��<C2Ӓ       �	�*��Yc�A�*

loss�'�=��\;       �	���Yc�A�*

lossH��=f�sW       �	�o��Yc�A�*

loss�W7>2p�Y       �	���Yc�A�*

loss]=4�q       �	����Yc�A�*

loss�|�=��w�       �	V��Yc�A�*

lossC�&=�85�       �	����Yc�A�*

loss�cv=���!       �	����Yc�A�*

loss��;��^       �	{3��Yc�A�*

loss�j�=�j       �	���Yc�A�*

losso�-=��i�       �	'j��Yc�A�*

loss��[=`�xF       �	���Yc�A�*

loss���=۵ɻ       �	ܡ��Yc�A�*

loss�H=d��       �	:��Yc�A�*

loss��r=3Za       �	Z���Yc�A�*

loss�i�=7�3B       �	�y��Yc�A�*

loss2g<�3�       �	 6��Yc�A�*

loss���<v���       �	;���Yc�A�*

lossX��=^ ��       �	���Yc�A�*

loss��;м��       �	� ��Yc�A�*

loss�މ=	k;S       �	����Yc�A�*

loss���=~�?�       �	�_��Yc�A�*

loss@�/>_y�+       �	�^��Yc�A�*

loss��~=�T�       �	���Yc�A�*

loss���<K�l�       �	q���Yc�A�*

lossQ�E<�"�k       �	�o��Yc�A�*

loss1S&=�1��       �	���Yc�A�*

lossm�=�&f       �	J���Yc�A�*

loss�T�<��5K       �	�M��Yc�A�*

loss��=K�pY       �	����Yc�A�*

lossz&�<;l�K       �	ĳ��Yc�A�*

lossĩC=+��       �		���Yc�A�*

loss�\�=d�5       �	H��Yc�A�*

loss}��<Ȓ�|       �	����Yc�A�*

lossA�=��\M       �		���Yc�A�*

loss�
�=�
SN       �	l!��Yc�A�*

lossDAl=��:       �	����Yc�A�*

loss�K=�rq�       �	�]��Yc�A�*

loss���<�4�z       �	0���Yc�A�*

lossi�< �S�       �	���Yc�A�*

lossN�o=*��       �	|)��Yc�A�*

loss��G=��F�       �	����Yc�A�*

loss��=�;��       �	�s��Yc�A�*

loss�j=��Q�       �	���Yc�A�*

lossI�=�Lx�       �	����Yc�A�*

loss�t=gl)�       �	�S��Yc�A�*

lossٓ�<����       �	���Yc�A�*

lossʬ�=c�4C       �	����Yc�A�*

loss���<+�+$       �	�,��Yc�A�*

loss�B/=���       �	*��Yc�A�*

loss�؀=)�J       �	Y���Yc�A�*

lossDE�=��t�       �	�C��Yc�A�*

lossM�#="�O&       �	����Yc�A�*

loss�_�=��Q�       �	t��Yc�A�*

loss��>}mx\       �	���Yc�A�*

loss?=So�       �	����Yc�A�*

loss��=6���       �	�X��Yc�A�*

loss�=Bx�_       �	Y1��Yc�A�*

lossН<@���       �	����Yc�A�*

loss���=؃       �	�d��Yc�A�*

loss�
>�,ރ       �	���Yc�A�*

loss��O=y�z       �	,���Yc�A�*

loss��:>D��       �	�5��Yc�A�*

loss�U�=!���       �	����Yc�A�*

losse�B>��#�       �	yt��Yc�A�*

lossJ=(=si       �	���Yc�A�*

lossQ��=�
�z       �	ݳ��Yc�A�*

loss��r=8W�       �	�V��Yc�A�*

loss��=�&�$       �	^���Yc�A�*

loss�D`=9Z�       �	=���Yc�A�*

loss<8�=nNW       �	�H �Yc�A�*

loss���=a��       �	�� �Yc�A�*

lossïB=��7!       �	���Yc�A�*

loss	�b=`z�       �	�5�Yc�A�*

lossh/	=Nr�0       �	^��Yc�A�*

loss���<l:݆       �	s�Yc�A�*

loss���<"Di       �	��Yc�A�*

loss�7F=��E       �	!��Yc�A�*

loss_C=��       �	�L�Yc�A�*

lossX>�_I       �	k��Yc�A�*

loss�[W=��       �	6��Yc�A�*

loss7 >���       �	�3�Yc�A�*

lossx�@=��       �	�p�Yc�A�*

loss��=}�       �	�	�Yc�A�*

loss��=[��D       �	P�	�Yc�A�*

loss�=x�0�       �	�@
�Yc�A�*

loss�(�=r� �       �	Q�
�Yc�A�*

loss�}=���5       �	Ou�Yc�A�*

lossC�=��       �	��Yc�A�*

loss;ۊ=?MF�       �	���Yc�A�*

loss���= ��       �	�j�Yc�A�*

loss�
=�Tq5       �	��Yc�A�*

loss8(�<�X;       �	���Yc�A�*

loss=���       �	n2�Yc�A�*

lossŋq=�-�K       �	W��Yc�A�*

loss�� >]��c       �	�p�Yc�A�*

loss��=�u�       �	>�Yc�A�*

loss�m�=�d7       �	��Yc�A�*

loss ��<��(       �	�A�Yc�A�*

loss��_<�^hH       �	���Yc�A�*

lossL��<�'x�       �	N��Yc�A�*

loss�HZ=f?f�       �	RF�Yc�A�*

lossW�=���       �	�N�Yc�A�*

loss��=����       �	W��Yc�A�*

loss���=�K��       �	4��Yc�A�*

loss�_=p.�t       �	K�Yc�A�*

loss.rE=��.       �	���Yc�A�*

loss��>(sa�       �	�s�Yc�A�*

loss�<�=i��       �	n��Yc�A�*

loss��=-��       �	 �Yc�A�*

lossd�<��e�       �	;��Yc�A�*

lossz��=��PV       �	�X�Yc�A�*

loss}�w=ᚿ;       �	z��Yc�A�*

loss� �=�$
c       �	��Yc�A�*

loss��=�       �	 �Yc�A�*

loss���=6�Q�       �	�!�Yc�A�*

loss;��<��%*       �	�!�Yc�A�*

loss�E=O��       �	}Z"�Yc�A�*

loss�h�=�n�       �	. #�Yc�A�*

loss���=���       �	�#�Yc�A�*

lossP�>R�X�       �	CW$�Yc�A�*

loss鶪=)ʬ       �	��$�Yc�A�*

loss�s<�2B       �	vm&�Yc�A�*

loss�� >�?�       �	5
'�Yc�A�*

loss��E=X�       �	�'�Yc�A�*

loss9<� j�       �	J(�Yc�A�*

loss�xq<��T       �	��(�Yc�A�*

loss�U�=�,       �	��)�Yc�A�*

lossos�<:!��       �	� *�Yc�A�*

lossfZ�=�*�       �	�*�Yc�A�*

loss�U�=-�&<       �	ur+�Yc�A�*

loss�b-=�yS       �	<,�Yc�A�*

lossl�u=��6a       �	|�,�Yc�A�*

loss�v�=����       �	�U-�Yc�A�*

loss�{�<[_��       �	��-�Yc�A�*

loss[�;Y�       �	М.�Yc�A�*

loss�7�=�3�       �	H/�Yc�A�*

loss#�5<�=�       �	4�/�Yc�A�*

lossN�_<�9|`       �	��0�Yc�A�*

loss��}<2,�j       �	�41�Yc�A�*

loss��=�=Q	       �	F�1�Yc�A�*

loss�	C=U]�7       �	�l2�Yc�A�*

loss��=2�5       �	�3�Yc�A�*

lossXh�=�V_       �	�3�Yc�A�*

loss��=1z��       �	Y44�Yc�A�*

loss\�3=�-��       �	��4�Yc�A�*

lossї�=�`j�       �	+�5�Yc�A�*

loss�:=F���       �	�?6�Yc�A�*

loss|�B=ge�       �	��6�Yc�A�*

lossx7=���       �	�{7�Yc�A�*

loss��>�d	�       �	m 8�Yc�A�*

loss�P<9�X|       �	��8�Yc�A�*

loss� >����       �	CV9�Yc�A�*

loss��=d
�7       �	)�9�Yc�A�*

lossOJt=��;�       �	n�:�Yc�A�*

loss6F^=���l       �	+;�Yc�A�*

lossp �=���       �	��;�Yc�A�*

loss�Y�=�e|�       �	�h<�Yc�A�*

lossH��=*�       �	�=�Yc�A�*

lossd�=wۡh       �	�=�Yc�A�*

loss�i�=V⒩       �	4G>�Yc�A�*

lossLq�=�^V�       �	m�>�Yc�A�*

loss��<Ȍ�       �	�E@�Yc�A�*

loss^9�=���O       �	��@�Yc�A�*

lossla�=Ha7�       �	L�A�Yc�A�*

loss��=ۃpZ       �	5*B�Yc�A�*

loss��=�;6�       �	��B�Yc�A�*

loss&��=n̍a       �	�hC�Yc�A�*

loss|=��f/       �	�D�Yc�A�*

lossϼ>H�	�       �	ӡD�Yc�A�*

lossim�=\o��       �	>AE�Yc�A�*

lossOI�<HF�       �	]�E�Yc�A�*

loss�!S=|�,       �	�xF�Yc�A�*

loss�Bg=hs�       �	�G�Yc�A�*

lossc�^=R"�       �	)�G�Yc�A�*

loss��=��F�       �	�KH�Yc�A�*

loss�VY=S�R       �	u�H�Yc�A�*

loss�I=���*       �	kI�Yc�A�*

loss]�c=v-�       �	�@J�Yc�A�*

lossԢ�=�A\       �	9�J�Yc�A�*

loss��X=St��       �	�K�Yc�A�*

loss_��=�b�-       �	�.L�Yc�A�*

loss:/�=	߀�       �	F�L�Yc�A�*

lossA:�=z� �       �	�yM�Yc�A�*

loss�
<�u�       �	vN�Yc�A�*

lossW=�x       �	ǸN�Yc�A�*

loss1�;=���       �	�UO�Yc�A�*

loss_�=�3��       �	w�O�Yc�A�*

loss6�V=���t       �	֐P�Yc�A�*

loss���>�       �	sQ�Yc�A�*

loss��=]+�j       �	�R�Yc�A�*

loss�/�<��L       �	'�R�Yc�A�*

loss�i�=F���       �	bS�Yc�A�*

loss�jM=�jX       �	&�S�Yc�A�*

loss�Q�=��6       �	�T�Yc�A�*

loss�[�=��       �	BAU�Yc�A�*

loss�]=ﾅ�       �	%V�Yc�A�*

loss2-=�3�       �	��V�Yc�A�*

loss�=�z;�       �	5~W�Yc�A�*

lossO�H=�V�       �	�X�Yc�A�*

loss���;r���       �	��X�Yc�A�*

loss�ي<��O�       �	�hY�Yc�A�*

loss$r�<�A�7       �		Z�Yc�A�*

losszo7=�)       �	��Z�Yc�A�*

loss�<�B�+       �	eS[�Yc�A�*

lossJ�=���R       �	r�[�Yc�A�*

losswQ�=�Lo�       �	��\�Yc�A�*

loss��!=�E5       �	8i]�Yc�A�*

lossl>�?�       �	�^�Yc�A�*

loss��=KM4�       �	��^�Yc�A�*

losso+�=�a�x       �	�C_�Yc�A�*

loss��=�d�       �	�_�Yc�A�*

loss��M=j�B�       �	�z`�Yc�A�*

lossm�p<Ⱥ�'       �	[(a�Yc�A�*

loss)n�=�4��       �	��a�Yc�A�*

loss]�[=����       �	Dob�Yc�A�*

loss.ά=��v       �	�*c�Yc�A�*

loss��?=�E#.       �	z�c�Yc�A�*

lossZ�==#�'�       �	��d�Yc�A�*

loss��=�Z,       �	JDe�Yc�A�*

loss=uG)B       �	��e�Yc�A�*

loss<�=�9�x       �	a�f�Yc�A�*

lossF�G=l�s�       �	�8g�Yc�A�*

loss�^U=��T       �	�g�Yc�A�*

loss���=\!I<       �	�h�Yc�A�*

loss*��=\q��       �	�;i�Yc�A�*

loss=]�=1�j       �	>�i�Yc�A�*

loss�0�<�#��       �	�j�Yc�A�*

loss�
�=����       �	z�k�Yc�A�*

loss���;���       �	�=l�Yc�A�*

loss�Kj=�.��       �	l�l�Yc�A�*

loss�Q�=�9�       �	�m�Yc�A�*

lossq7�=�:�       �	Pn�Yc�A�*

loss�4H=�=��       �	��n�Yc�A�*

lossa�=�SP       �	�Yp�Yc�A�*

loss�*�=ѩ��       �	�q�Yc�A�*

loss���=n���       �	��q�Yc�A�*

loss4f�=����       �	=`r�Yc�A�*

lossR�c=�\��       �	�s�Yc�A�*

loss4i)=`���       �	��s�Yc�A�*

loss��=�x6�       �	ert�Yc�A�*

loss.�w<��[       �	|&u�Yc�A�*

loss
=��S�       �	J�u�Yc�A�*

lossc��<�h'       �	A�v�Yc�A�*

lossĴb=^��n       �	�1w�Yc�A�*

loss~#=���E       �	e�w�Yc�A�*

loss��?=�F��       �	8�x�Yc�A�*

loss��B==z�       �	4Gy�Yc�A�*

loss�"�<�U��       �	)�y�Yc�A�*

lossͿ�<�Wl�       �	��z�Yc�A�*

loss�w�=���       �	t@{�Yc�A�*

loss���=���z       �	|�Yc�A�*

loss؁�=���0       �	��|�Yc�A�*

lossfVw=aS�       �	G}�Yc�A�*

lossDb�=��0       �	�O~�Yc�A�*

lossv��=Z�ų       �	`�Yc�A�*

loss�n�=��?�       �	���Yc�A�*

lossoqc=��{�       �	n��Yc�A�*

loss��=Z���       �	���Yc�A�*

loss#>��z       �	H���Yc�A�*

lossh�=���v       �	�i��Yc�A�*

lossI�;n�NE       �	�H��Yc�A�*

loss��<�>i       �	���Yc�A�*

loss ��=LPա       �	���Yc�A�*

loss�QL>�m��       �	�ǈ�Yc�A�*

loss��n=��.        �	�5��Yc�A�*

losss̿<�'       �	�*��Yc�A�*

loss,�=��I       �	$Ԑ�Yc�A�*

loss�9>yL��       �	�z��Yc�A�*

lossg&�=W��       �	B#��Yc�A�*

loss=�>N�1       �	���Yc�A�*

loss�>3=i�e        �	�ʓ�Yc�A�*

loss�Z�=�Y       �	�~��Yc�A�*

loss�A�=U4{R       �	�,��Yc�A�*

loss�n=��ș       �	�ޕ�Yc�A�*

loss��=����       �	a���Yc�A�*

loss��<�X��       �	G:��Yc�A�*

lossV�<9���       �	���Yc�A�*

loss�<=�V�       �	����Yc�A�*

loss:W�=�O$�       �	f���Yc�A�*

loss���<���
       �	I��Yc�A�*

loss�i =G|�       �	_��Yc�A�*

loss�<+���       �	5���Yc�A�*

loss

=H@t}       �	gF��Yc�A�*

loss��T=���K       �	g��Yc�A�*

lossI>����       �	����Yc�A�*

lossZ��=�4R'       �	�E��Yc�A�*

lossݙ�=Vэy       �	 ��Yc�A�*

loss�=Y��       �	���Yc�A�*

loss�V<����       �	y@��Yc�A�*

loss�Ͻ=�
p�       �	\ ��Yc�A�*

loss��T>q B       �	��Yc�A�*

loss�~=���       �	qU��Yc�A�*

lossԌ8=?��       �	����Yc�A�*

loss���=�N       �	���Yc�A�*

lossT��=SM�       �	����Yc�A�*

loss�=j�       �	 V��Yc�A�*

loss��4=���       �	���Yc�A�*

loss��=���       �	ӄ��Yc�A�*

lossT�9>���       �	���Yc�A�*

loss&ɟ=ya�       �	t���Yc�A�*

losso��=/k��       �	<P��Yc�A�*

loss�[<=���-       �	���Yc�A�*

loss�#r=�NY       �	E���Yc�A�*

loss;��=�ST       �	)"��Yc�A�*

loss��L=��4       �	ۿ��Yc�A�*

loss���<t}
       �	p`��Yc�A�*

loss���<.G z       �	r���Yc�A�*

loss���=����       �	����Yc�A�*

lossB�=��*       �	=G��Yc�A�*

loss&�I=��       �	A���Yc�A�*

loss��=��ŏ       �	���Yc�A�*

loss�=~�א       �	�B��Yc�A�*

lossW*==���       �	~��Yc�A�*

loss9�<A),~       �	����Yc�A�*

loss#�=b��       �	j2��Yc�A�*

loss��<�fQ�       �	Ա�Yc�A�*

lossRb�=3� �       �	o��Yc�A�*

loss�K>RO�{       �	I��Yc�A�*

lossi1w=�KL       �	`w��Yc�A�*

loss
O<i�	e       �	���Yc�A�*

loss���<�ț�       �	&���Yc�A�*

lossݽ=���       �	�G��Yc�A�*

loss��/=�HT       �	���Yc�A�*

lossZ��=�(       �	����Yc�A�*

loss��{>S       �	�4��Yc�A�*

loss�`�=�w%]       �	kԸ�Yc�A�*

loss�N=�c#�       �	!���Yc�A�*

loss�M >|���       �	xF��Yc�A�*

loss#�=ho'|       �	Hݺ�Yc�A�*

loss�=J���       �	v��Yc�A�*

lossH�=�m�       �	���Yc�A�*

loss��8=�̤       �	ڭ��Yc�A�*

loss��<�)�       �	S��Yc�A�*

lossmP�=�Z        �	x��Yc�A�*

lossC=T���       �	J���Yc�A�*

loss��7=�(�       �	�6��Yc�A�*

loss���<�au�       �	���Yc�A�*

lossd9V=x#cn       �	S���Yc�A�*

loss�y�<f��       �	U��Yc�A�*

lossH��<|o�R       �	g���Yc�A�*

loss#>ؾ��       �	����Yc�A�*

loss,[=e��       �	�%��Yc�A�*

lossťi=)��       �	Y���Yc�A�*

loss�:=z`;       �	I���Yc�A�*

loss��=�y��       �	E.��Yc�A�*

loss�̍<+v�M       �	���Yc�A�*

loss�=�P-�       �	�b��Yc�A�*

loss�\k=��       �	���Yc�A�*

loss��=}�       �	����Yc�A�*

loss�<�<$eKw       �	kG��Yc�A�*

loss�[�=���       �	���Yc�A�*

loss;��<�'�+       �	����Yc�A�*

loss���=p]S       �	\��Yc�A�*

loss���=��M�       �	���Yc�A�*

loss���<�g�       �	����Yc�A�*

loss�A=��       �	-B��Yc�A�*

losspF�<ᯤt       �	����Yc�A�*

loss�R=�ڱ�       �	τ��Yc�A�*

loss,L�=�I5�       �	?7��Yc�A�*

loss.�=_\t       �	����Yc�A�*

lossϠ=�A       �	p��Yc�A�*

loss�[=+�v
       �	���Yc�A�*

loss��=�RU       �	1���Yc�A�*

loss2@�=~�m�       �	�M��Yc�A�*

loss&I<=��	O       �	����Yc�A�*

lossre�=kXK       �	ҋ��Yc�A�*

loss|<vk�       �	I��Yc�A�*

loss}u9=��O�       �	:���Yc�A�*

loss��C=. ;       �	���Yc�A�*

lossl�p<1��~       �	C ��Yc�A�*

loss_[,=��8�       �	���Yc�A�*

loss&��=@�4I       �	*���Yc�A�*

loss��Q>���X       �	Qf��Yc�A�*

loss��<�L��       �	r���Yc�A�*

loss�� =�޽       �	����Yc�A�*

lossZ�=r�^       �	ly��Yc�A�*

losswC�<���Y       �		��Yc�A�*

loss��=.�t7       �	���Yc�A�*

loss�N�<;EB�       �	���Yc�A�*

lossҼ�<
��       �	�^��Yc�A�*

loss�7�<�n�       �	A���Yc�A�*

loss�ۻ<v�f       �	���Yc�A�*

loss?gI<t�~`       �	�'��Yc�A�*

loss�� =&�r       �	r���Yc�A�*

loss���<� 4-       �	�p��Yc�A�*

loss (;:�#�       �	���Yc�A�*

loss�)<�S��       �	.���Yc�A�*

lossR=T>�$       �	o��Yc�A�*

loss\q�<{
J       �	���Yc�A�*

lossX�=On%P       �	u���Yc�A�*

loss���;j��       �	�K��Yc�A�*

loss�\=QQï       �	����Yc�A�*

lossH��>��x       �	���Yc�A�*

lossNp�<-rF7       �	?��Yc�A�*

losszT�<�l˾       �	w���Yc�A�*

loss�V�=��L�       �	Jy��Yc�A�*

lossN��=b��       �	���Yc�A�*

loss�9^=���n       �	���Yc�A�*

lossM"n=�i�       �	Qg��Yc�A�*

loss�ӻ=g���       �	��Yc�A�*

lossW%�=�]�v       �	���Yc�A�*

loss�Z=��2�       �	����Yc�A�*

lossx�2>1��       �	J���Yc�A�*

loss!4=�Y$�       �	�4��Yc�A�*

loss�]�=~яF       �	A���Yc�A�*

loss+N�=�\�\       �	nm��Yc�A�*

loss�+(=����       �	x��Yc�A�*

lossm�=���       �	���Yc�A�*

lossӖ=x�G       �	 ���Yc�A�*

loss�E4=�yU@       �	�K��Yc�A�*

loss+�=���w       �	i���Yc�A�*

lossݾ�=�x�       �	����Yc�A�*

loss��$=}�       �	B$��Yc�A�*

loss���<U��       �	����Yc�A�*

loss,$�<�d�       �	.p��Yc�A�*

lossx�6=mf�P       �	>��Yc�A�*

loss��<����       �	9���Yc�A�*

lossȥ�=/{�       �	LQ��Yc�A�*

lossa�<�p+       �	!��Yc�A�*

loss�՜<�5�       �	����Yc�A�*

loss��4=ES�P       �	�O��Yc�A�*

lossk=l�;H       �	���Yc�A�*

loss���=�U\J       �	�M��Yc�A�*

loss�I�<�y�y       �	����Yc�A�*

loss��+=��O       �	����Yc�A�*

lossmY>=!��4       �	�)��Yc�A�*

lossHG<��       �	����Yc�A�*

loss�/=��M       �	�k��Yc�A�*

loss�=%f��       �	���Yc�A�*

lossȉ1<�%/�       �	,���Yc�A�*

loss�?=/��       �	6Z��Yc�A�*

loss��=a3��       �	����Yc�A�*

loss�f=�kp�       �	���Yc�A�*

loss�|:=L0ˇ       �	4 �Yc�A�*

loss��L=Ѱ�m       �	c� �Yc�A�*

loss�l�<��u�       �	�e�Yc�A�*

loss
"X=\��       �	v��Yc�A�*

loss� �<'��       �	���Yc�A�*

loss��=h�       �	1�Yc�A�*

loss��6=DDf�       �	��Yc�A�*

loss���<
��6       �	Ϡ�Yc�A�*

loss��=�5!u       �	q9�Yc�A�*

lossMc�<�k��       �	���Yc�A�*

loss��L<����       �	3o�Yc�A�*

loss�Yd=�l�;       �	�z�Yc�A�*

loss���=9�<�       �	� �Yc�A�*

loss�Sy=~�y]       �	s� �Yc�A�*

loss�>�=<[L       �	�!�Yc�A�*

loss���=4E�       �	0"�Yc�A�*

loss.�!=z�       �	��"�Yc�A�*

loss�ʑ=�C�       �	�k#�Yc�A�*

loss�ee= f��       �	=($�Yc�A�*

loss�N�=�u�N       �	?�$�Yc�A�*

lossf�l=>�74       �	�d%�Yc�A�*

loss���<>4�       �	�%�Yc�A�*

loss�A=��D�       �	�&�Yc�A�*

loss[=H�+�       �	�Y'�Yc�A�*

lossVy=����       �	��'�Yc�A�*

loss=�=��}       �	��(�Yc�A�*

loss�U�=�"�       �	�G)�Yc�A�*

loss��I;�2��       �	��)�Yc�A�*

lossS?�=�^4�       �	��*�Yc�A�*

loss��=���       �	.+�Yc�A�*

loss\`>�M�       �	��+�Yc�A�*

loss�T=��g�       �	Ί,�Yc�A�*

loss�R=�n}       �	,*-�Yc�A�*

loss�?=+U@�       �	m�-�Yc�A�*

lossͭ�=�ve+       �	�l.�Yc�A�*

loss{�<Wh�       �	E/�Yc�A�*

loss�Ñ=ٮ��       �	O�/�Yc�A�*

loss
�L=w%��       �	�O0�Yc�A�*

loss���<��c�       �	5�0�Yc�A�*

loss��h=����       �	�1�Yc�A�*

lossq�	=0�       �	�$2�Yc�A�*

loss���<=.�       �	��2�Yc�A�*

loss%��<�܂C       �	�h3�Yc�A�*

lossm��<�Qt       �	�	4�Yc�A�*

loss�+�=��       �	��4�Yc�A�*

loss�=��@�       �	�t5�Yc�A�*

loss��=�e�&       �	6�Yc�A�*

loss��<�8�       �	��6�Yc�A�*

loss��O=ba7       �	�e7�Yc�A�*

loss���=~���       �	,8�Yc�A�*

loss���={]�       �	}�8�Yc�A�*

loss�"J=M�       �	UQ9�Yc�A�*

loss���=쒞�       �	��9�Yc�A�*

loss4=_'cT       �	-�:�Yc�A�*

loss
�a=�7�G       �	;8;�Yc�A�*

loss=��=V1��       �	��;�Yc�A�*

losso�=�b�0       �	 s<�Yc�A�*

lossc�=�_��       �	�=�Yc�A�*

loss��=_�{       �	c�=�Yc�A�*

loss��Y=��a       �	HS>�Yc�A�*

loss�
<NLu       �	'�>�Yc�A�*

lossz<t<��        �	,�?�Yc�A�*

loss�?=o�`       �	KV@�Yc�A�*

loss]��<Y5'�       �	/�@�Yc�A�*

loss[�7>{�9{       �	�A�Yc�A�*

loss�4�<.h�_       �	�@B�Yc�A�*

loss}��<��ƌ       �	7�B�Yc�A�*

loss�^	=ٸ�       �	�C�Yc�A�*

loss�#=��.�       �	N%D�Yc�A�*

loss�e=t�I       �	'�D�Yc�A�*

lossoR=��g       �	wdE�Yc�A�*

loss|��=Zkհ       �	>F�Yc�A�*

loss9ϛ<igA       �	{�F�Yc�A�*

lossШ<y��       �	�?G�Yc�A�*

loss@C0=���<       �	��G�Yc�A�*

loss�hA=.�_       �	�wH�Yc�A�*

loss�ׯ<��mV       �	W'I�Yc�A�*

loss)�[=66�B       �	f�I�Yc�A�*

loss�o�=��{       �	�tJ�Yc�A�*

loss=�=��       �	bK�Yc�A�*

loss=�=�XKJ       �	g�K�Yc�A�*

loss�V/=a6
�       �	`L�Yc�A�*

loss��/=���       �	/M�Yc�A�*

loss���<��       �	��M�Yc�A�*

loss�2
=��'       �	��N�Yc�A�*

loss4	=�O/       �	�7O�Yc�A�*

lossu��=+�=�       �	Y�O�Yc�A�*

lossLp�="c�       �	i�P�Yc�A�*

loss��=�`"�       �	�7Q�Yc�A�*

loss6��<���y       �	��Q�Yc�A�*

loss?��=��       �	k~R�Yc�A�*

lossF{�="��       �	�cS�Yc�A�*

lossx��=ٕy_       �	GT�Yc�A�*

loss*]�=�
       �	��T�Yc�A�*

loss�/=e^K       �	9CU�Yc�A�*

loss{}=�k�       �	1�U�Yc�A�*

loss�=�x"I       �	ŏV�Yc�A�*

loss��1=��>Y       �	�.W�Yc�A�*

loss�]�=�nv�       �	��W�Yc�A�*

loss�X=��d�       �	ԁX�Yc�A�*

loss��<�'wD       �		7Y�Yc�A�*

loss�>L�N&       �	A�Y�Yc�A�*

lossJ�=��k       �	�vZ�Yc�A�*

loss&:=.�)       �	�[�Yc�A�*

loss�s�=��J�       �	��[�Yc�A�*

lossԅ>s���       �	SZ\�Yc�A�*

loss���<F���       �	��\�Yc�A�*

lossn�=L��       �	��]�Yc�A�*

loss�=uk!D       �	�8^�Yc�A�*

loss�P>Wy��       �	��^�Yc�A�*

lossQh=u���       �	o�_�Yc�A�*

loss���<�n�F       �	�/`�Yc�A�*

lossF�<hR�<       �	|�`�Yc�A�*

loss��> Wu6       �	!sa�Yc�A�*

loss��<\|\�       �	�b�Yc�A�*

loss{Z=��"�       �	�b�Yc�A�*

loss�Ee=��%�       �	]Sc�Yc�A�*

lossz�<��       �	Dd�Yc�A�*

loss:#=�U8t       �	Z�d�Yc�A�*

loss�.
>>H�A       �	�Xe�Yc�A�*

loss��=߉��       �	��e�Yc�A�*

loss���=�%��       �	��f�Yc�A�*

loss��c=�Pp|       �	�Dg�Yc�A�*

loss��5=�\f�       �	��g�Yc�A�*

lossl
�<{�\|       �	|h�Yc�A�*

loss��l=�ڜ       �	. i�Yc�A�*

loss�o8=�,�>       �	{�i�Yc�A�*

loss,L�=�؂       �	�Sj�Yc�A�*

loss��u=���        �	��j�Yc�A�*

lossX��=4b*�       �	��k�Yc�A�*

lossd�E<�y       �	,l�Yc�A�*

lossC�<=�(;       �	�m�Yc�A�*

loss�ٕ=���{       �	/�m�Yc�A�*

lossF4�=��
k       �	On�Yc�A�*

loss���<�ur       �	��n�Yc�A�*

loss�>A=9/K(       �	��o�Yc�A�*

loss��=��@       �	 bp�Yc�A�*

lossDM>�T��       �	�q�Yc�A�*

loss �6=���!       �	?�q�Yc�A�*

lossoʤ={NC�       �	�?r�Yc�A�*

lossJC�=ֲ]�       �	��r�Yc�A�*

loss]�
=Y�7Q       �	4�s�Yc�A�*

lossL=���-       �	�;t�Yc�A�*

loss�=#7�       �	��t�Yc�A�*

loss�[�<�»�       �	�u�Yc�A�*

loss���= ���       �	�^v�Yc�A�*

loss4��=��pe       �	5_w�Yc�A�*

loss]L=���       �	��w�Yc�A�*

loss_U1=2ǝ�       �	�x�Yc�A�*

loss4b�<���       �	�y�Yc�A�*

loss�r9=X\%�       �	i�z�Yc�A�*

loss;�v<0�Q       �	�-{�Yc�A�*

loss��=���`       �	��{�Yc�A�*

loss26�=?䑳       �	4h|�Yc�A�*

loss�=5/�n       �	��}�Yc�A�*

loss\�=_       �	�~�Yc�A�*

loss?�=��       �	���Yc�A�*

loss�ؿ={��       �	=��Yc�A�*

lossvC�<?�u�       �	Ś�Yc�A�*

loss���<1�s�       �	I���Yc�A�*

loss��<�}K�       �	y>��Yc�A�*

loss8�	=�u?       �	���Yc�A�*

loss�ۊ=}�T�       �	9~��Yc�A�*

loss>��=it�$       �	b��Yc�A�*

lossԯ�=�Fx�       �	մ��Yc�A�*

loss�=� �       �	XY��Yc�A�*

losss<C<y1��       �	��Yc�A�*

loss�h=��v�       �	(���Yc�A�*

losslZU=d�       �	{f��Yc�A�*

lossk�=�Hd       �	���Yc�A�*

loss�&=�*       �	ٴ��Yc�A�*

loss�}�=Ҕ�       �	�U��Yc�A�*

loss�D�<;z��       �	8���Yc�A�*

loss�=��       �	���Yc�A�*

loss�	<���P       �	*��Yc�A�*

loss��=D���       �	`ʍ�Yc�A�*

loss�xS=�6       �	�h��Yc�A�*

lossr��<ߠ+�       �	h��Yc�A�*

loss\�q=�t��       �	����Yc�A�*

loss�O/=�	D)       �	�N��Yc�A�*

loss=F=jQ       �	���Yc�A�*

loss�d�=�j�0       �	����Yc�A�*

loss�R=*�       �	*��Yc�A�*

loss���<k��o       �	;�Yc�A�*

loss@kZ=y�y       �	�g��Yc�A�*

loss�
�<�dA8       �	�.��Yc�A�*

loss��F=A��6       �	�ʔ�Yc�A�*

lossM�>��DV       �	�f��Yc�A�*

loss��=��U       �	C��Yc�A�*

lossw�Q=�]QF       �	����Yc�A�*

lossA�<͞m�       �	�u��Yc�A�*

loss��.<���       �	���Yc�A�*

lossd,�=�w�W       �	a���Yc�A�*

loss�3h<����       �	JB��Yc�A�*

loss���=�E�       �	Qݙ�Yc�A�*

lossjn�<#��n       �	�v��Yc�A�*

loss��<�{/0       �	d��Yc�A�*

loss��@=��/�       �	H���Yc�A�*

loss�5�<� �       �	F���Yc�A�*

loss4J�<}%        �	�5��Yc�A�*

loss�և=��'�       �	M؝�Yc�A�*

loss�%>"7�=       �	dw��Yc�A�*

lossStP=�ENB       �	���Yc�A�*

loss�f6=Sۿ�       �	r���Yc�A�*

losslC�<��g�       �	�=��Yc�A�*

loss-��<$�֋       �	�٠�Yc�A�*

loss	�=$RƜ       �	���Yc�A�*

loss)�=�r�       �	���Yc�A�*

loss�=R ��       �	h���Yc�A�*

loss��=Gj?       �	F��Yc�A�*

loss�V1=~��:       �	+ޣ�Yc�A�*

loss̥n=`���       �	!w��Yc�A�*

lossqL�<9�&�       �	$��Yc�A�*

loss9=��       �	����Yc�A�*

loss� =k�S       �	!X��Yc�A�*

lossn�=G�C�       �	M���Yc�A�*

lossH�;=�       �	����Yc�A�*

loss�"�=9��c       �	�$��Yc�A�*

loss�B*={���       �	����Yc�A�*

loss#YM=��e       �	d[��Yc�A�*

loss�|=p���       �	(���Yc�A�*

loss�s:=+4��       �	L���Yc�A�*

loss,��<�{�        �	,���Yc�A�*

loss�W=���~       �	G��Yc�A�*

loss�M<4�B�       �	g��Yc�A�*

loss~=5T��       �	߿��Yc�A�*

lossl��=c���       �	�b��Yc�A�*

loss!�x=E�4       �	s��Yc�A�*

loss�e>_Z�P       �	p���Yc�A�*

lossq �=�s�J       �	�_��Yc�A�*

lossh�1>))F       �	���Yc�A�*

loss�ݔ<��/j       �	����Yc�A�*

lossg=�	        �	kE��Yc�A�*

loss3�=��oO       �	Ō��Yc�A�*

loss���=`hT       �	�'��Yc�A�*

loss��=~�Wt       �	 ���Yc�A�*

loss�?�=f���       �	����Yc�A�*

lossԚ�=�?h       �	�V��Yc�A�*

loss�Q�=W��       �	���Yc�A�*

loss�m_<(?7>       �	k���Yc�A�*

loss�4=�
�       �	�V��Yc�A�*

loss�~;=�9X       �	����Yc�A�*

loss\�=Z���       �	���Yc�A�*

loss�Z�=���       �	�=��Yc�A�*

loss4�]=��԰       �	�ں�Yc�A�*

loss	M�=�w��       �	����Yc�A�*

lossc�o=Ѷ_�       �	� ��Yc�A�*

loss�Ky=�[�       �	V���Yc�A�*

loss�E�=���       �	r���Yc�A�*

loss�ܣ=��&�       �	����Yc�A�*

loss�<=ۉM       �	�<��Yc�A�*

loss}
M<���       �	c��Yc�A�*

loss*0;=OEg�       �	�+��Yc�A�*

loss��]=c�Z�       �	v���Yc�A�*

loss�ű=��/       �	����Yc�A�*

loss.Dp=ю�       �	�8��Yc�A�*

loss���<�Lڧ       �	�O��Yc�A�*

lossO�m=w�n�       �	j���Yc�A�*

losslۂ=�5       �	����Yc�A�*

lossM��=���       �	�@��Yc�A�*

loss{-=��V�       �	����Yc�A�*

loss�(L=�B\       �	����Yc�A�*

lossa�>Rj�r       �	�?��Yc�A�*

loss��J=\��       �	����Yc�A�*

loss�Hp=�q�       �	�v��Yc�A�*

lossd�j;90|M       �	���Yc�A�*

loss��=�t+       �	����Yc�A�*

loss�8�=�@��       �	Qj��Yc�A�*

loss
yd=m��E       �	���Yc�A�*

loss}�E=�}       �	ӡ��Yc�A�*

lossQy=&�N       �	�P��Yc�A�*

loss���<Vhc       �	����Yc�A�*

lossWp�<�Nqh       �	����Yc�A�*

loss��=�V       �	�2��Yc�A�*

loss�/Y=yPF       �	K���Yc�A�*

losso�E=��¯       �	(c��Yc�A�*

loss��=��3�       �	���Yc�A�*

loss�ʦ=oNE       �	����Yc�A�*

lossI=�Ð	       �	2V��Yc�A�*

lossF�>x�       �	����Yc�A�*

loss^�=��?�       �	���Yc�A�*

lossx�=N�:       �	SA��Yc�A�*

loss_4=|&/       �	����Yc�A�*

loss�2�=%<�i       �	�r��Yc�A�*

lossOo=�g"       �	���Yc�A�*

loss`{>�o�       �	q���Yc�A�*

loss�M�=0m��       �	7���Yc�A�*

loss$,�=W�2       �	�Q��Yc�A�*

loss�u/=�}V       �	��Yc�A�*

loss���=D[       �	���Yc�A�*

loss琊=Q݈>       �	�;��Yc�A�*

loss�i�;��#       �	Y���Yc�A�*

loss�KS<C��       �	v��Yc�A�*

loss��T=�{�A       �	e��Yc�A�*

loss���<�N^�       �	f���Yc�A�*

loss�8�=�|��       �	)\��Yc�A�*

loss=�x=B�hm       �	���Yc�A�*

loss?�<��ئ       �	����Yc�A�*

lossl�&=-%[       �	8���Yc�A�*

loss��=q�(�       �	$��Yc�A�*

loss�iJ<\,N
       �	���Yc�A�*

loss��D<5.d,       �	�W��Yc�A�*

loss��=6��       �	����Yc�A�*

loss$!�<�oAP       �	����Yc�A�*

loss�x�<�>       �	l[��Yc�A�*

loss8X�<��	       �	����Yc�A�*

loss�w�=*�=L       �	����Yc�A�*

loss�~�<�9�       �	�]��Yc�A�*

loss�$�=K0�       �	%��Yc�A�*

loss�օ=3���       �	����Yc�A�*

loss��
>h�ۏ       �	�=��Yc�A�*

lossR=ӝC=       �	����Yc�A�*

loss�=�H��       �	����Yc�A�*

loss��=�^�i       �	�V��Yc�A�*

lossϐ�=`&'C       �	����Yc�A�*

loss7#B=�`ڌ       �	���Yc�A�*

loss�7l=Li       �	q��Yc�A�*

loss�x<���       �	����Yc�A�*

loss�<=��k       �	�J��Yc�A�*

loss���=�k��       �	a���Yc�A�*

lossI��=iW�       �	`w��Yc�A�*

lossܵ�=g7       �	o��Yc�A�*

loss�T�=����       �	$���Yc�A�*

loss�)=��؝       �	�M��Yc�A�*

loss2G=(�       �	h���Yc�A�*

loss�4�<�(�       �	{���Yc�A�*

loss��v=E���       �	�H��Yc�A�*

loss�!=#�p@       �	����Yc�A�*

loss\�-=X��o       �	����Yc�A�*

lossѬk=�[��       �	�a��Yc�A�*

loss�%�=�i��       �	?���Yc�A�*

loss���<��2       �	����Yc�A�*

loss���=oh�C       �	�I��Yc�A�*

losse=&��n       �	����Yc�A�*

loss�e<�/@       �	
���Yc�A�*

loss��g=�#�d       �	`��Yc�A�*

loss(>�/q�       �	���Yc�A�*

lossh|�=d*!       �	nQ��Yc�A�*

loss��<trIQ       �	����Yc�A�*

loss�=(��       �	'���Yc�A�*

lossN�l=�"�       �	C��Yc�A�*

loss�$=���2       �	����Yc�A�*

loss)})=�G�       �	(c��Yc�A�*

loss���=l��K       �	���Yc�A�*

loss=Ń<`Co�       �	����Yc�A�*

loss�C�=���       �	a���Yc�A�*

lossw�=�0�       �	����Yc�A�*

loss�G�=�u�+       �	JE �Yc�A�*

loss��8=�F��       �	�� �Yc�A�*

loss3�O=���       �	<��Yc�A�*

loss�8i<l��       �	hu�Yc�A�*

lossNI=�D�       �	���Yc�A�*

loss�*�<��       �	�j�Yc�A�*

loss��x=Ƙ@�       �	�D�Yc�A�*

losspM=��a�       �	.��Yc�A�*

lossv!Z>�*w       �	j��Yc�A�*

lossF��=.�m�       �	;��Yc�A�*

lossn��=smrI       �	��	�Yc�A�*

loss1e�=֊��       �	�=
�Yc�A�*

loss���<���       �	u�
�Yc�A�*

loss���=0��n       �	���Yc�A�*

loss���< Sb1       �	z�Yc�A�*

lossSu=��p       �	��Yc�A�*

loss�\�<���       �	���Yc�A�*

loss�=!�Y       �	�p�Yc�A�*

lossv"�=���       �	��Yc�A�*

loss���<N�S       �	��Yc�A�*

loss�r(=�d)p       �	�H�Yc�A�*

loss�3�<5Й7       �	y �Yc�A�*

loss�<K�z�       �	Z��Yc�A�*

loss�V%=�o��       �	���Yc�A�*

loss8X�=�7�       �	'�Yc�A�*

loss�.=���       �	S��Yc�A�*

loss
s�=��       �	�j�Yc�A�*

loss�C�=����       �	K!�Yc�A�*

lossM��=��D�       �	���Yc�A�*

loss�>�=�C��       �	2Y�Yc�A�*

loss]M�=����       �	
M�Yc�A�*

loss;!�=�E�       �	��Yc�A�*

loss1/=�i�       �	Ҍ�Yc�A�*

loss�WV<���       �	�*�Yc�A�*

loss��5=����       �	Z��Yc�A�*

lossq�q=c�       �	���Yc�A�*

lossFBw=��       �	D�Yc�A�*

lossd�T<m	��       �	&��Yc�A�*

lossvgh=Uk��       �	؀�Yc�A�*

losss�>P�DY       �	� �Yc�A�*

lossI�X=??��       �	���Yc�A�*

lossؗq=O���       �	�U�Yc�A�*

loss��%=G�V       �	s��Yc�A�*

loss��=	s�       �	��Yc�A�*

lossD=��3       �	@0 �Yc�A�*

lossc �=��ȶ       �	h� �Yc�A�*

lossO܌<�R�.       �	��!�Yc�A�*

loss�گ<8k8r       �	g"�Yc�A�*

loss�8!<rf       �	-
#�Yc�A�*

loss}�5=�,�       �	)�#�Yc�A�*

loss̵h=�E	       �	ݕ$�Yc�A�*

loss�J�=i��X       �	�2%�Yc�A�*

loss=�u=�C��       �	3�%�Yc�A�*

loss觋<xo~       �	1{&�Yc�A�*

lossV|J='r�       �	�'�Yc�A�*

loss�m=D�.       �	˻'�Yc�A�*

loss�}�=G�R       �	�W(�Yc�A�*

loss��=�td7       �	Z�(�Yc�A�*

loss�ё<7i�O       �	��)�Yc�A�*

loss�">0��l       �	=*�Yc�A�*

lossa�<�p       �	��*�Yc�A�*

loss�0=�6�#       �	��+�Yc�A�*

loss]Շ=�	F�       �	�5,�Yc�A�*

lossU�=.XK�       �	��,�Yc�A�*

loss,g<�w�       �	w�-�Yc�A�*

loss&�>�w       �	�W.�Yc�A�*

loss2�4=�>6�       �	�*/�Yc�A�*

loss}��<J��       �	=�/�Yc�A�*

loss106=�+L       �	&p0�Yc�A�*

loss�OM=r���       �	�1�Yc�A�*

loss���=M�Df       �	��1�Yc�A�*

loss7ۑ=)b-2       �	P2�Yc�A�*

loss��=9��       �	��2�Yc�A�*

loss Q�=,է�       �	��3�Yc�A�*

lossߥ~=����       �	V�4�Yc�A�*

loss��=X�8�       �	�;5�Yc�A�*

loss���=���       �	�76�Yc�A�*

loss�q">�fW�       �	��6�Yc�A�*

loss�	�=ҝP@       �	Z~7�Yc�A�*

loss�L+>��C�       �	�8�Yc�A�*

loss��<�6^       �	y�8�Yc�A�*

loss��=�%�;       �	�P9�Yc�A�*

loss�_=#�k:       �	�:�Yc�A�*

loss
�u=��9b       �	�:�Yc�A�*

loss`��=���       �	�q;�Yc�A�*

losss_s=��+�       �	�<�Yc�A�*

lossȎ�<��l       �	��<�Yc�A�*

losss�>�h�       �	�\=�Yc�A�*

lossn5�=v�R3       �	��=�Yc�A�*

loss���=��       �	-?�Yc�A�*

loss�ds=AJ�J       �	��?�Yc�A�*

losse�=���       �	�@�Yc�A�*

loss�^E=ݫ�N       �	s+A�Yc�A�*

loss��|<ٛf�       �	^�A�Yc�A�*

loss(!�=_d�       �	ЗB�Yc�A�*

loss���=+?��       �	�<C�Yc�A�*

lossҝ<L�Ń       �	��C�Yc�A�*

loss��=j��X       �	S�D�Yc�A�*

loss.�Z=fk<       �	�AE�Yc�A�*

loss}=l�B�       �	��E�Yc�A�*

loss�M=k��       �	Q�F�Yc�A�*

loss2�$=�       �	�G�Yc�A�*

loss�4=.�7b       �	&�G�Yc�A�*

loss���<#|u       �	p_H�Yc�A�*

losssD�=��#I       �	�H�Yc�A�*

loss
�=t0       �	��I�Yc�A�*

loss<s=�CH6       �	wMJ�Yc�A�*

loss�$=�w5�       �	<�J�Yc�A�*

loss��<��cE       �	��K�Yc�A�*

loss��b=��׭       �	"RL�Yc�A�*

lossͶ5>[���       �	� M�Yc�A�*

loss]��=���       �	��M�Yc�A�*

loss�=�S�       �	�SN�Yc�A�*

loss��=�@�9       �	��N�Yc�A�*

loss�l�=9��       �	��O�Yc�A�*

loss&�'=|��       �	>=P�Yc�A�*

loss�~�<\        �	w�P�Yc�A�*

losszj�=R�       �	�uR�Yc�A�*

loss}��=�Y�P       �	I�S�Yc�A�*

lossl-�=<eʗ       �	�'T�Yc�A�*

loss���=B���       �	��T�Yc�A�*

loss�=O!��       �	�U�Yc�A�*

loss�
=\�d       �	�QV�Yc�A�*

loss]B=2&�       �	=�V�Yc�A�*

loss�4= ��f       �	?�W�Yc�A�*

loss	�<���(       �	�4X�Yc�A�*

loss��<�=7       �	�Y�Yc�A�*

loss;��=W��       �	m7Z�Yc�A�*

lossQ��=�xR       �	��Z�Yc�A�*

loss%��<��       �	؞[�Yc�A�*

loss�z`=fO��       �	C\�Yc�A�*

loss>�<1�2a       �	��\�Yc�A�*

loss&@=�L��       �	��]�Yc�A�*

loss��<�ju2       �	1&^�Yc�A�*

loss��<h߱       �	D�^�Yc�A�*

loss�=p�       �	W`_�Yc�A�*

lossȂ�=M�xJ       �	��_�Yc�A�*

loss��=��2�       �	͕`�Yc�A�*

loss݆�=��)       �	2a�Yc�A�*

loss܏�<�1�1       �	K�a�Yc�A�*

loss��<��E       �	Mgb�Yc�A�*

loss7S=Â?       �	_c�Yc�A�*

loss�C�<f��@       �	��c�Yc�A�*

lossH�=���T       �	�Id�Yc�A�*

loss���=e,�       �	7�d�Yc�A�*

loss�c�=.}�       �	�~e�Yc�A�*

loss#�=*���       �	f�Yc�A�*

loss�b�=㱆Y       �	>�f�Yc�A�*

loss�=�db       �	�Ng�Yc�A�*

lossi�/=��       �	1�g�Yc�A�*

loss��=���V       �	��h�Yc�A�*

loss��=ϔ�0       �	�2i�Yc�A�*

lossɪ=e�Y       �	�i�Yc�A�*

loss���=���       �	�uj�Yc�A�*

loss��@=xb�       �	�k�Yc�A�*

loss[u,=v���       �	��k�Yc�A�*

loss�1�<Bi�\       �	0Ll�Yc�A�*

loss�A"=��
       �	h�l�Yc�A�*

loss�<s%�1       �	i�m�Yc�A�*

loss8=��ή       �	�(n�Yc�A�*

loss���=�`v�       �	��n�Yc�A�*

loss�9=;#nO       �	Hjo�Yc�A�*

loss	V=�jV�       �	|p�Yc�A�*

loss���=�k�(       �	��p�Yc�A�*

lossW�?=�ᡆ       �	ZFq�Yc�A�*

loss�,�<�-�       �	i�q�Yc�A�*

lossdo<�>       �	ρr�Yc�A�*

loss��<��3       �	
+s�Yc�A�*

loss$�=�J�k       �	>�s�Yc�A�*

loss��=���       �	�vt�Yc�A�*

loss)n=02��       �	Qu�Yc�A�*

loss�[=��[       �	K�u�Yc�A�*

loss�V�=e9�C       �	�Vv�Yc�A�*

loss���=W�r�       �	��v�Yc�A�*

loss(M�<����       �	��w�Yc�A�*

loss�Dj=`U�       �	�px�Yc�A�*

loss�	�<ć�       �	�y�Yc�A�*

lossR=[`��       �	�y�Yc�A�*

loss��=�:*;       �	�Sz�Yc�A�*

lossa��=��       �	0�z�Yc�A�*

losss�=��3       �	~�{�Yc�A�*

loss�*�=Z��       �	R*|�Yc�A�*

lossΞ#=�=�7       �	��|�Yc�A�*

loss�+�=WB       �	��}�Yc�A�*

lossф<+��)       �	��~�Yc�A�*

losso�=k���       �	*;�Yc�A�*

loss8�<(�h       �	���Yc�A�*

loss�``=~���       �	/���Yc�A�*

loss���<KAO       �	�D��Yc�A�*

loss���<��X�       �	U݁�Yc�A�*

lossWkM=nD�       �	�x��Yc�A�*

loss徾=���       �	���Yc�A�*

loss&�=i�       �	t���Yc�A�*

loss�E�<J�x        �	t^��Yc�A�*

loss}�l<2��       �	����Yc�A�*

loss\�=di��       �	.���Yc�A�*

lossAs\<���n       �	�)��Yc�A�*

lossN֏;9��H       �	j���Yc�A�*

lossx�u<A�9i       �	U��Yc�A�*

loss���<���j       �	���Yc�A�*

loss
j;��-       �	����Yc�A�*

loss�r =�v3       �	g+��Yc�A�*

loss:�<�^kK       �	�ȉ�Yc�A�*

loss�p�=o��       �	~t��Yc�A�*

loss,��;n��       �	���Yc�A�*

lossd�;�C�       �	\���Yc�A�*

lossS��:��       �	�I��Yc�A�*

loss,,w<�M�       �	���Yc�A�*

loss���=m�k"       �	V��Yc�A�*

loss$�=w�?�       �	���Yc�A�*

losstL�;���       �	����Yc�A�*

loss]0<�|]a       �	LO��Yc�A�*

loss�4p>��|�       �	����Yc�A�*

loss�~<s,c�       �	����Yc�A�*

loss��O<��~        �	�h��Yc�A�*

lossOFm=?��m       �	9��Yc�A�*

lossc��=q$�4       �	����Yc�A�*

lossk=���g       �	-B��Yc�A�*

loss,}�<�45       �	&��Yc�A�*

loss���=�9��       �	���Yc�A�*

loss�Ӫ=��H�       �	�F��Yc�A�*

lossz�=�j�       �	&��Yc�A�*

loss@�=K3P�       �	�{��Yc�A�*

loss�O�<U �       �	�'��Yc�A�*

loss��9=����       �	�Ǘ�Yc�A�*

loss�=Ն�       �	og��Yc�A�*

lossӜ�=��}       �		��Yc�A�*

loss��=���       �	�Й�Yc�A�*

lossBQ>��d�       �	�n��Yc�A�*

lossD��<IFW       �	q;��Yc�A�*

lossv�%=��ٝ       �	{1��Yc�A�*

loss>�>�!�J       �	����Yc�A�*

lossDQ<��eR       �	{���Yc�A�*

lossq�+<#Q%       �	A*��Yc�A�*

lossx��<�`       �	h̞�Yc�A�*

loss*�+=�5͖       �	�l��Yc�A�*

loss:�<M�d�       �	$
��Yc�A�*

lossw��<��IM       �	���Yc�A�*

loss-Z�;U�2       �	�?��Yc�A�*

loss�?
=N�Z       �	ߡ�Yc�A�*

loss�Y=�2��       �	Wx��Yc�A�*

loss�J=.�t%       �	"3��Yc�A�*

loss)�4=	���       �	Kˣ�Yc�A�*

loss���<l\��       �	�l��Yc�A�*

loss%yv=��       �	���Yc�A�*

loss�I<�SI�       �	����Yc�A�*

lossuh;���       �	����Yc�A�*

loss}�`=�cz�       �	#0��Yc�A�*

loss?��<��~@       �	�է�Yc�A�*

loss�)<�j�U       �	�r��Yc�A�*

lossJ�=!�       �	i��Yc�A�*

loss&4.=��       �	g���Yc�A�*

loss  �<����       �	a��Yc�A�*

lossb�<��C_       �	*���Yc�A�*

loss9[#=ߎ�<       �	����Yc�A�*

loss��=+��       �	�@��Yc�A�*

loss� =|�2�       �	Qܬ�Yc�A�*

lossF�v<�Y��       �	b���Yc�A�*

lossA&7=�>|Z       �	�5��Yc�A�*

loss�j=z�k�       �	}ͮ�Yc�A�*

loss�E�<~��       �	�d��Yc�A�*

loss�Ib=����       �	����Yc�A�*

loss�q%=Y�`       �	,���Yc�A�*

loss�n=�"�       �	�u��Yc�A�*

loss�[�==�       �	�#��Yc�A�*

losss�t=К�g       �	����Yc�A�*

loss�,�=Q�n�       �	-���Yc�A�*

loss�.�=���       �	�?��Yc�A�*

loss�)�==�j�       �	����Yc�A�*

loss=#�=t!��       �	�~��Yc�A�*

loss�t�=R�x�       �	���Yc�A�*

loss_{�=
�3       �	"���Yc�A�*

loss�I�=�g1       �	RC��Yc�A�*

loss�!_=�B�       �	����Yc�A�*

loss��:<�u
�       �	n��Yc�A�*

loss�I=AQ(�       �	E��Yc�A�*

lossOE=w�p�       �	���Yc�A�*

loss�޴=^D?>       �	 T��Yc�A�*

loss^='Q�L       �	����Yc�A�*

lossT0>�k"S       �	ˆ��Yc�A�*

loss��<�}��       �	�.��Yc�A�*

loss w=�BP       �	���Yc�A�*

loss�=g       �	����Yc�A�*

loss@-�=���       �	�:��Yc�A�*

lossvҡ<ZA�       �	����Yc�A�*

lossG5�=/�b       �	jm��Yc�A�*

loss���<� �G       �	h��Yc�A�*

lossoN�=�lL�       �	����Yc�A�*

lossܳ4=K��       �	�>��Yc�A�*

loss���<�gd�       �	����Yc�A�*

lossn�&=��2       �	?r��Yc�A�*

loss=�=3��       �	���Yc�A�*

lossf$0=�~       �	����Yc�A�*

loss}�4=}3�&       �	�g��Yc�A�*

loss��<=�vTy       �	7 ��Yc�A�*

loss�F=�7�@       �	ۢ��Yc�A�*

lossCR�<�Ɋ       �	�_��Yc�A�*

loss���=B�d�       �	���Yc�A�*

loss�b�<���       �	���Yc�A�*

loss��E=��R�       �	<��Yc�A�*

loss�*w;�ٓ�       �	����Yc�A�*

lossoJ�=!�?c       �	am��Yc�A�*

loss�t>��h�       �	���Yc�A�*

loss�+�=�q@j       �	����Yc�A�*

loss��
=GΏ        �	�S��Yc�A�*

loss�/�=ZI�       �	���Yc�A�*

loss!�D<ݚ^D       �	ú��Yc�A�*

lossvƇ=��	       �	DR��Yc�A�*

loss�6�=M2�       �	@���Yc�A�*

loss :0=���       �	����Yc�A�*

lossyL=Ch\       �	�2��Yc�A�*

losst��<tm       �	J���Yc�A�*

loss�F�<7k�       �	�r��Yc�A�*

loss�M�<�~��       �	F��Yc�A�*

lossXI�<�(��       �	5���Yc�A�*

loss��=���       �	�N��Yc�A�*

loss�9�<�]:       �	q���Yc�A�*

loss?B>7�       �	�{��Yc�A�*

loss��g=���G       �	���Yc�A�*

loss�p<�{�f       �	̴��Yc�A�*

loss,`�:��J/       �	����Yc�A�*

loss��V<H�`�       �	.��Yc�A�*

loss̄P<�^       �	5���Yc�A�*

loss��=����       �	Yj��Yc�A�*

losspH�=O��       �	o��Yc�A�*

lossO#�<�y/       �	%���Yc�A�*

loss[h�<S(�       �	PP��Yc�A�*

lossȠ�=�v�       �	����Yc�A�*

loss��<��n       �	=���Yc�A�*

loss�A�;�]v       �	���Yc�A�*

loss�}y=���m       �	����Yc�A�*

loss�ߺ<�?ZZ       �	�j��Yc�A�*

loss��=����       �	O��Yc�A�*

lossF��=Ƽ�       �	a���Yc�A�*

loss`d�=�|��       �	�=��Yc�A�*

loss�? =�jn)       �	���Yc�A�*

loss�h
=$�'       �	ux��Yc�A�*

lossm�D=�rG�       �	���Yc�A�*

loss�P=B�       �	����Yc�A�*

losssH=��       �	�I��Yc�A�*

loss�=.?z�       �	����Yc�A�*

loss́i=��Jq       �	����Yc�A�*

loss���<T��}       �	'� �Yc�A�*

loss�:�<��j?       �	&9�Yc�A�*

loss��[=���)       �	���Yc�A�*

loss�s�==Ϲd       �	�n�Yc�A�*

loss��	>�x�       �	��Yc�A�*

loss��=�n��       �	Ǡ�Yc�A�*

loss[�<����       �	�v�Yc�A�*

lossF�<�3�       �	���Yc�A�*

loss�=�#ė       �	p�Yc�A�*

lossw�<l�'       �		�Yc�A�*

loss�=_�       �	��Yc�A�*

loss�H	=��x       �	�X�Yc�A�*

loss���=S)ݛ       �	#��Yc�A�*

lossT�=��S�       �	��	�Yc�A�*

loss���<���?       �	L3
�Yc�A�*

loss{�=���       �	�
�Yc�A�*

loss���=�!��       �	�d�Yc�A�*

loss��<!��       �	�Yc�A�*

lossF|�<����       �	G��Yc�A�*

loss�?=��i       �	�N�Yc�A�*

loss_�:>2���       �	���Yc�A�*

loss���=��0�       �	���Yc�A�*

loss*l�<�h�       �	�A�Yc�A�*

loss��<��(�       �	���Yc�A�*

loss�4�=���       �	L��Yc�A�*

loss�}x=E4��       �	$%�Yc�A�*

lossS�g=��"       �	��Yc�A�*

loss%b>�j��       �	�X�Yc�A�*

loss�A�<Dt\�       �	���Yc�A�*

loss���<�L�o       �	e��Yc�A�*

loss]F>�S/       �	+�Yc�A�*

loss�=t�       �	p��Yc�A�*

loss�ӛ=�b]�       �	�e�Yc�A�*

loss�_�=:8Do       �	��Yc�A�*

loss�H=�_u�       �	:��Yc�A�*

lossq'=��k�       �	�W�Yc�A�*

lossE��=N{       �	���Yc�A�*

lossxT�<[�ٰ       �	e��Yc�A�*

loss�&�=�@�       �	ʣ�Yc�A�*

loss� D='��       �	5{�Yc�A�*

loss�a�<l��       �	M�Yc�A�*

loss7;l<N�       �	��Yc�A�*

loss��d=��       �	�F�Yc�A�*

loss��T=����       �	f��Yc�A�*

loss��=��=�       �	w�Yc�A�*

loss��=�S�       �	|�Yc�A�*

loss$?z<[9�z       �	��Yc�A�*

loss�I=��v�       �	�;�Yc�A�*

loss�g<��       �	_��Yc�A�*

loss�ֆ=����       �	�k �Yc�A�*

loss�E�=��)       �	d!�Yc�A�*

loss�>TN�       �	��!�Yc�A�*

loss��=���.       �	�<"�Yc�A�*

loss�Q=�Ʌ       �	��"�Yc�A�*

loss!<����       �	Po#�Yc�A�*

losss$�<7Jn�       �	B$�Yc�A�*

lossq<P=:x�h       �	=�$�Yc�A�*

loss!�=)@��       �	%;%�Yc�A�*

loss�F�<��       �	&�Yc�A�*

loss8��<�m�s       �	*�&�Yc�A�*

lossi=Zz{�       �	�?'�Yc�A�*

loss�=g�p       �	�'�Yc�A�*

loss�_<�z �       �	��(�Yc�A�*

loss���<Q        �	�G)�Yc�A�*

loss�=�=�M       �	��)�Yc�A�*

loss �,=L�&       �	��*�Yc�A�*

lossN�=��       �	gC+�Yc�A�*

loss���<3e       �	��+�Yc�A�*

loss�R=(�L       �	ǀ,�Yc�A�*

loss��E<9�ɗ       �	9*-�Yc�A�*

loss��)=���[       �	�-�Yc�A�*

lossOT>Z���       �	#j.�Yc�A�*

lossԮ�<|=`       �	�/�Yc�A�*

loss�x=�P�-       �	��/�Yc�A�*

loss��=�(PR       �	AI0�Yc�A�*

loss=[=�hB       �	��0�Yc�A�*

loss�G�=��9       �		�1�Yc�A�*

loss߾^<)�$�       �	� 2�Yc�A�*

loss��<�nH�       �	��2�Yc�A�*

loss/u=��s       �	eT3�Yc�A�*

loss�}�=kV��       �	N+4�Yc�A�*

loss}*+=ٕD       �	��4�Yc�A�*

loss4�W=��2u       �	�n5�Yc�A�*

lossQ�1=!���       �	�
6�Yc�A�*

loss���=�	�	       �	�6�Yc�A�*

lossw�3<&��       �	�>7�Yc�A�*

loss���<Wh�       �	��7�Yc�A�*

lossH�=&�X�       �	w8�Yc�A�*

loss6Z=Y~��       �	�9�Yc�A�*

loss;�<�Ҽ       �	:�9�Yc�A�*

loss`6=�g;       �	�H:�Yc�A�*

loss�;�<_t��       �	��:�Yc�A�*

loss���=��-�       �	��;�Yc�A�*

loss�Z<S�@�       �	�<�Yc�A�*

lossV�.<�ϓ1       �	�<�Yc�A�*

lossn�6=^���       �	.T=�Yc�A�*

loss�;=���L       �	��=�Yc�A�*

loss�=�C�       �	�>�Yc�A�*

loss2��=m��       �	�!?�Yc�A�*

loss1E9=3t	�       �	��?�Yc�A�*

loss��B=�       �	�@�Yc�A�*

lossw�<U�e       �	)]A�Yc�A�*

lossct{<�-�       �	�B�Yc�A�*

lossrէ=��       �	��B�Yc�A�*

loss��=#E]	       �	WC�Yc�A�*

loss��<G&`�       �	b�C�Yc�A�*

loss@ճ<G�&�       �	[�D�Yc�A�*

loss>\<U���       �	UjE�Yc�A�*

lossj5=�.��       �	�F�Yc�A�*

loss���<�F��       �	�6G�Yc�A�*

loss��S=ˉ��       �	��G�Yc�A�*

loss	If=��(:       �	��H�Yc�A�*

loss �.=�d�       �	7I�Yc�A�*

loss��#=�rq'       �	�I�Yc�A�*

loss��<���C       �	eUJ�Yc�A�*

loss���<!*}       �	��J�Yc�A�*

loss��G<۸@q       �	k�K�Yc�A�*

lossE�4=Xx�       �	V,L�Yc�A�*

loss��G=�y��       �	��L�Yc�A�*

loss��=l�h       �	~sM�Yc�A�*

loss=�:=�I��       �	�N�Yc�A�*

lossj��<��@{       �	g�N�Yc�A�*

loss�G�=��v�       �	ȖO�Yc�A�*

loss�W='��,       �	\:P�Yc�A�*

loss#��<Ӆ[&       �	��P�Yc�A�*

lossna,=G1�       �	��Q�Yc�A�*

loss��<�A��       �	�ZR�Yc�A�*

loss%.=����       �	��R�Yc�A�*

lossHd=q��,       �	��S�Yc�A�*

loss|��=lFb       �	fMT�Yc�A�*

loss߁�<���       �	��T�Yc�A�*

losse��=���       �	;�U�Yc�A�*

loss�Y={X�9       �	�V�Yc�A�*

loss��.=��       �	w0W�Yc�A�*

loss)s"=��Q�       �	��W�Yc�A�*

loss�Ht<!�+�       �	�`X�Yc�A�*

loss
p=�4{�       �	M�X�Yc�A�*

lossv�=�R1�       �	��Y�Yc�A�*

lossݴ�<w��<       �	�,Z�Yc�A�*

loss�VH>�ڒ�       �	��Z�Yc�A�*

loss���=h�0U       �	�[�Yc�A�*

loss<yc=��%       �	�9\�Yc�A�*

loss�W�<�nE       �	}�\�Yc�A�*

loss�<C��       �	�b]�Yc�A�*

lossp��=��       �	��]�Yc�A�*

loss�;>o��       �	y�^�Yc�A�*

lossv
�=\�m�       �	�3_�Yc�A�*

loss��=�
       �	�_�Yc�A�*

lossm��=���       �	t`�Yc�A�*

loss��u=Z��       �	La�Yc�A�*

loss�6==�\��       �	-�a�Yc�A�*

loss�<���       �	�Nb�Yc�A�*

loss��<��4�       �	��b�Yc�A�*

loss16�<r,       �	�c�Yc�A�*

loss��)=Q})L       �	Yd�Yc�A�*

loss�;=��       �	E�d�Yc�A�*

loss�D�=]BX       �	ŏe�Yc�A�*

lossW�=�H�;       �	�+f�Yc�A�*

loss#ޮ=8�!�       �	��f�Yc�A�*

lossV\A=)/��       �	'hg�Yc�A�*

loss�j�=�/�d       �	�
h�Yc�A�*

loss�<GM�f       �	�h�Yc�A�*

loss���<��|�       �	�Mi�Yc�A�*

loss��T=� �       �	��i�Yc�A�*

loss6�=t�Vc       �	��j�Yc�A�*

lossv�=� �=       �	ak�Yc�A�*

loss�.:=y���       �	��k�Yc�A�*

loss?��<���K       �	�l�Yc�A�*

loss�9�<7�a       �	�,m�Yc�A�*

loss���<���1       �	6�m�Yc�A�*

loss�(=��       �	tan�Yc�A�*

loss>_=�݃       �	eo�Yc�A�*

loss�kq=�X�       �	�o�Yc�A�*

loss�Y�=ٻM�       �	qup�Yc�A�*

loss�=�Sf       �	q�Yc�A�*

loss��<.؉�       �	�q�Yc�A�*

loss!nH<����       �	u[r�Yc�A�*

loss�9=�O�       �	�r�Yc�A�*

loss�=t*��       �	�s�Yc�A�*

loss���<�t�       �	�_t�Yc�A�*

loss�6�<�-L�       �	@�t�Yc�A�*

lossf=���&       �	1�u�Yc�A�*

lossd�(=�hj�       �	�=v�Yc�A�*

loss��<0S)       �	M�v�Yc�A�*

loss�t=�K       �	�tw�Yc�A�*

loss���=n\�;       �	� x�Yc�A�*

loss,��=��       �	��x�Yc�A�*

loss��=���       �	why�Yc�A�*

loss�=�=��       �	�z�Yc�A�*

loss��,=�p�i       �	��z�Yc�A�*

loss)�=.m��       �	�Z{�Yc�A�*

lossz��=P,+       �	��{�Yc�A�*

loss���<�1gR       �	E�|�Yc�A�*

loss
AY={�X       �	�=}�Yc�A�*

loss��=��       �	��}�Yc�A�*

loss_)R=�x\7       �	�y~�Yc�A�*

lossf/=���       �	��Yc�A�*

lossn��=�H��       �	��Yc�A�*

lossc�=�T{�       �	LQ��Yc�A�*

lossG�<K�Il       �	�U��Yc�A�*

loss�_�=��g�       �	���Yc�A�*

lossƗ�=�R�       �	���Yc�A�*

lossc4�<=m�X       �	���Yc�A�*

loss�|=�+/>       �	F���Yc�A�*

loss�C=����       �	*���Yc�A�*

loss8=�<�h�       �	�<��Yc�A�*

loss��=��7*       �	Z���Yc�A�*

loss��=K~       �	�É�Yc�A�*

lossE�=�!P       �	����Yc�A�*

lossZ��<=+W�       �	����Yc�A�*

lossT;=       �	*���Yc�A�*

lossr��<<G       �	����Yc�A�*

loss��;�(��       �	�F��Yc�A�*

lossO֜=��V       �	[ӏ�Yc�A�*

lossq�<���       �	<���Yc�A�*

loss�p�<�6q�       �	�ő�Yc�A�*

loss�ԙ=Z��5       �	Wv��Yc�A�*

lossX4�=/d�<       �	<���Yc�A�*

loss�R�<O�H       �	�Yc�A�*

loss-�4=�Σ�       �	�Y��Yc�A�*

lossDF�=����       �	O<��Yc�A�*

loss&�=� ��       �	g)��Yc�A�*

loss�	�<||�R       �	w+��Yc�A�*

lossK�=����       �	%Θ�Yc�A�*

losswC|=���       �	�k��Yc�A�*

loss��= �;�       �	U���Yc�A�*

lossGu=6~�       �	h���Yc�A�*

lossaY~=�\�%       �	w���Yc�A�*

loss�u<���a       �	�5��Yc�A�*

loss�|='���       �	}ʝ�Yc�A�*

loss�=="��       �	�a��Yc�A�*

loss�*�=)�G       �	d���Yc�A�*

loss�a(=j�fV       �	�6��Yc�A�*

loss��=���*       �		��Yc�A�*

lossw��=~�U�       �	���Yc�A�*

loss=51�V       �	ƿ��Yc�A�*

lossט�<~�       �	�s��Yc�A�*

loss�Vc=�:/|       �	����Yc�A�*

loss���<����       �	3��Yc�A�*

loss��<[���       �	�ͥ�Yc�A�*

loss��=Xx`       �	�f��Yc�A�*

loss<�<���$       �	���Yc�A�*

loss�:=��/�       �	ӟ��Yc�A�*

lossP�=r_��       �	7��Yc�A�*

loss��%=��׶       �	�ͨ�Yc�A�*

loss�'=!�[       �	�l��Yc�A�*

loss���=+�       �	��Yc�A�*

loss�@=�Ḝ       �	����Yc�A�*

loss��=�<�       �	7Q��Yc�A�*

loss�=����       �	s���Yc�A�*

loss��=�_L�       �	����Yc�A�*

lossk�=�@��       �	3��Yc�A�*

loss��< n��       �	���Yc�A�*

lossb�<�~       �	�z��Yc�A�*

loss��;�76       �	���Yc�A�*

loss��=��l       �	���Yc�A�*

loss���=�I�       �	,~��Yc�A�*

loss�=����       �	ͱ�Yc�A�*

loss.�O=��2       �	�p��Yc�A�*

loss�C4=�1�W       �	
��Yc�A�*

loss��M=�<џ       �	�³�Yc�A�*

loss#Q<�:+       �	�c��Yc�A�*

losss�9=�]7�       �	����Yc�A�*

loss��<s-q�       �	囵�Yc�A�*

loss�-E=,�       �	�4��Yc�A�*

lossE��<�Y�6       �	(Ӷ�Yc�A�*

losss�6>9a�w       �	�n��Yc�A�*

loss�>�=;fnU       �	/��Yc�A�*

loss���<��       �	���Yc�A�*

loss
޼=8E��       �	�Q��Yc�A�*

loss.�F=�*%       �	+���Yc�A�*

loss�2�=��^�       �	ꖺ�Yc�A�*

loss)��<Oe�C       �	<��Yc�A�*

loss@L�<�W       �	Eջ�Yc�A�*

loss3'Q<$bF�       �	s��Yc�A�*

loss/Q�=��       �	� ��Yc�A�*

loss_�=�ª       �	���Yc�A�*

lossa <��       �	�c��Yc�A�*

loss�~�<C���       �	���Yc�A�*

loss[�)=_���       �	���Yc�A�*

loss�3�=�6r�       �	o���Yc�A�*

loss���=|�       �	����Yc�A�*

loss��g=�d�A       �	a7��Yc�A�*

lossߔ�<"3�       �	h���Yc�A�*

loss���=���b       �	���Yc�A�*

lossM.L="l�       �	d"��Yc�A�*

lossAɀ=�oc�       �	O���Yc�A�*

loss�o�=��       �	e��Yc�A�*

lossn�=�f�       �	$��Yc�A�*

lossJw�<N]       �	`���Yc�A�*

loss��=��&l       �	���Yc�A�*

lossE.�<יu       �	5%��Yc�A�*

loss���<�E       �	s���Yc�A�*

loss�<:Bq       �	�M��Yc�A�*

loss!s=ӄ�G       �	s���Yc�A�*

loss�?�<���z       �	7���Yc�A�*

loss�+�=t��       �	�1��Yc�A�*

loss�_�=���       �	����Yc�A�*

loss�=v�ߥ       �	yt��Yc�A�*

loss���<`6�L       �	���Yc�A�*

lossd)0=2@*       �	"���Yc�A�*

loss�i�=1��       �	#L��Yc�A�*

lossA.=�m�       �	d��Yc�A�*

loss�w=��%�       �	Û��Yc�A�*

loss��<]�
       �	[D��Yc�A�*

loss{��=�+9Z       �	a���Yc�A�*

loss<�ǧK       �	o~��Yc�A�*

lossh�-<4�$"       �	a��Yc�A�*

loss��=_��       �	����Yc�A�*

loss�ub=<p�/       �	�U��Yc�A�*

loss��<��       �	����Yc�A�*

loss@�'=��:       �	G���Yc�A�*

lossz%=�e       �	�&��Yc�A�*

loss�9`=�u�       �	h��Yc�A�*

loss!!=��O       �	����Yc�A�*

loss�= G�*       �	u>��Yc�A�*

loss� =d�r       �	r���Yc�A�*

lossS�=X��       �	H���Yc�A�*

loss��@=�h�       �	 $��Yc�A�*

loss�^t<x�       �	]���Yc�A�*

loss��<s*I�       �	�e��Yc�A�*

loss��= �.[       �	G��Yc�A�*

loss���<b��:       �	���Yc�A�*

loss���<Wv>I       �	�@��Yc�A�*

loss�==y�;�       �	����Yc�A�*

loss[=�Jm       �	9{��Yc�A�*

lossz��<E���       �	���Yc�A�*

loss	$=:w��       �	����Yc�A�*

loss�#�=��C�       �	RG��Yc�A�*

loss_;=Jxw�       �	����Yc�A�*

loss�=i��V       �	���Yc�A�*

loss�:`=��q       �	� ��Yc�A�*

loss�4=�\�Q       �	����Yc�A�*

loss�s�=؟�	       �	���Yc�A�*

loss,��<�4L;       �	g���Yc�A�*

loss�=�)       �	�[��Yc�A�*

loss��=3��       �	%��Yc�A�*

loss_'�=���b       �	���Yc�A�*

lossE&<a=G       �	�:��Yc�A�*

loss�&s<�2�~       �	y���Yc�A�*

loss�_�=����       �	M���Yc�A�*

loss�1�=�H       �	6��Yc�A�*

loss�Z5=��       �	|���Yc�A�*

loss&lm=�r       �	Qi��Yc�A�*

loss��j<n�X&       �	�	��Yc�A�*

loss��=�7m       �	\���Yc�A�*

loss�|D=l�       �	�b��Yc�A�*

loss��=���       �	3���Yc�A�*

losssPy=m��\       �	���Yc�A�*

loss_d�=MG.       �	�9��Yc�A�*

lossl�'=���       �	����Yc�A�*

loss =���       �	���Yc�A�*

loss�B=�eL
       �	4M��Yc�A�*

loss��=��x       �	K���Yc�A�*

loss�Į<Xa�,       �	n���Yc�A�*

loss&k�= T��       �	�0��Yc�A�*

lossj��= �J�       �	>���Yc�A�*

loss��<�j�       �	�e��Yc�A�*

loss�r�<MfG       �	u ��Yc�A�*

loss��V=)�j       �	����Yc�A�*

loss���<L��       �	80��Yc�A�*

loss��<��(�       �	`���Yc�A�*

loss,	�=I�S�       �	�`��Yc�A�*

lossn��=~��&       �	Q���Yc�A�*

loss�E�=�+Gw       �	d���Yc�A�*

loss�O:={\5�       �	4,��Yc�A�*

loss~�=#��       �	����Yc�A�*

loss��h=���#       �	}Z��Yc�A�*

lossc>�\       �	8���Yc�A�*

loss�1�=����       �	���Yc�A�*

loss�a=���       �	�@��Yc�A�*

loss��S=�X>_       �	����Yc�A�*

loss��<=�h!       �	Cq��Yc�A�*

loss:s�<'��        �	1��Yc�A�*

loss A�<��:        �	���Yc�A�*

loss`�R=H	�       �	;5��Yc�A�*

loss�Q>��0       �	����Yc�A�*

loss{�=>�&O       �	fj �Yc�A�*

loss��^=�ް*       �	�>�Yc�A�*

loss�}�<o� B       �	��Yc�A�*

loss��<��X�       �	��Yc�A�*

loss��8=$�k       �	�-�Yc�A�*

losst,@<�·r       �	o��Yc�A�*

loss֘�<P��Y       �	���Yc�A�*

loss���<��       �	�b�Yc�A�*

lossCr=�G       �	#��Yc�A�*

loss�$@=O6s       �	�T�Yc�A�*

loss�	N=Mc�L       �	M��Yc�A�*

loss;��;�r�g       �	Y�	�Yc�A�*

losso�=�H�       �	;
�Yc�A�*

loss��<��T       �	)�
�Yc�A�*

lossrh�<3���       �	Ts�Yc�A�*

loss|�=�6       �	��Yc�A�*

lossi�<x�br       �	��Yc�A�*

loss�A.=8�N{       �	�Q�Yc�A�*

lossԒ�=��)       �	���Yc�A�*

loss��=�C��       �	<��Yc�A�*

loss��|<����       �	;�Yc�A�*

lossh��<*;��       �	 ��Yc�A�*

loss�=�=���-       �	'P�Yc�A�*

loss���<�5��       �	���Yc�A�*

lossRO*>�h2;       �	'��Yc�A�*

loss�DO>E���       �	�Yc�A�*

loss3X=K�S�       �	��Yc�A�*

loss��>=`�fx       �	�D�Yc�A�*

lossN�=u�J�       �	���Yc�A�*

loss<j==.C�       �	?q�Yc�A�*

lossD�<��v4       �	p�Yc�A�*

loss!l�;K�g       �	^��Yc�A�*

loss6MP=�=�g       �	�D�Yc�A�*

loss�?�<��5       �	���Yc�A�*

loss�W=5x�:       �	�o�Yc�A�*

loss�� =h>T�       �	y�Yc�A�*

loss�_�<��       �	צ�Yc�A�*

lossOͮ;F"       �	�W�Yc�A�*

loss��<fZ-       �	[��Yc�A�*

loss�O=��<�       �	D�Yc�A�*

loss l�<�-]       �	���Yc�A�*

loss��a=9E�V       �	�x�Yc�A�*

loss[$=��e�       �	{1�Yc�A�*

loss�a=��'�       �	}��Yc�A�*

loss���=I��       �	�p�Yc�A�*

loss��=���       �	��Yc�A�*

loss)�
=�2b4       �	+��Yc�A�*

loss�y�<~,D       �	�G �Yc�A�*

lossz�n<��J�       �	�!�Yc�A�*

loss8=���N       �	�!�Yc�A�*

losso��<��	�       �	XW"�Yc�A�*

loss�e5=eG�       �	/�"�Yc�A�*

lossQ�+=�Ge�       �	��#�Yc�A�*

loss-�
=�?q       �	�k$�Yc�A�*

lossf��=5�Fa       �	�%�Yc�A�*

loss�=�<ex�       �	��%�Yc�A�*

lossg�=\	��       �	C&�Yc�A�*

loss�?�<�7|       �	/�&�Yc�A�*

loss{U�=ѻ�b       �	9{'�Yc�A�*

loss�@C=è��       �	�&(�Yc�A�*

loss��<0D
�       �	��(�Yc�A�*

lossrf`=��?m       �	��)�Yc�A�*

loss��<�ap       �	[@*�Yc�A�*

loss��=��       �	^�*�Yc�A�*

loss���=��b       �	@�+�Yc�A�*

loss�3�<à�        �	�(,�Yc�A�*

lossv$=VC��       �	��,�Yc�A�*

lossNl�;�C       �	By-�Yc�A�*

loss\�A=a]�_       �	�.�Yc�A�*

loss
�3=§N       �	ض.�Yc�A�*

lossJׅ<j�c       �	Y/�Yc�A�*

lossT��<d�j�       �	�/�Yc�A�*

loss�_=c�c�       �	��0�Yc�A�*

loss*"z=���}       �	�(1�Yc�A�*

loss�Y,=<QY       �	 �1�Yc�A�*

loss�y�<Hѩ�       �	Ab2�Yc�A�*

loss�yj=r��       �	�3�Yc�A�*

loss��b<�~�%       �	Q�3�Yc�A�*

loss��D;@�!       �	�x4�Yc�A�*

loss�;i<*�:X       �	�5�Yc�A�*

loss8��<B��       �	�5�Yc�A�*

loss�9�<�y       �	/R6�Yc�A�*

lossW8�<�x��       �	� 7�Yc�A�*

loss�7�;	��       �	�7�Yc�A�*

loss�(=��       �	�U8�Yc�A�*

loss"��<��       �	�
9�Yc�A�*

loss�ݍ:$�b�       �	4�9�Yc�A�*

loss�<6�m'       �	�B:�Yc�A�*

loss���;i���       �	3�:�Yc�A�*

loss�q�<�+�       �	�v;�Yc�A�*

lossj��<���A       �	9<�Yc�A�*

lossi)`;��<�       �	T�<�Yc�A�*

loss�#�;�3�       �	x=�Yc�A�*

loss,͟>�鋌       �	�>�Yc�A�*

loss���:1m٬       �	5�>�Yc�A�*

lossJUb<AK�       �	nN?�Yc�A�*

loss1 7=��El       �	�?�Yc�A�*

loss��f=���V       �	�@�Yc�A�*

loss䙈<�M       �	qA�Yc�A�*

losse��<��       �	Z�A�Yc�A�*

lossr�m=����       �	��B�Yc�A�*

loss#�B=�       �	�C�Yc�A�*

loss�x=큈       �	]�D�Yc�A�*

loss���=d�A       �	tBF�Yc�A�*

loss��=ξ��       �	��F�Yc�A�*

loss�=���       �	{G�Yc�A�*

loss��=R	P       �	ϼH�Yc�A�*

loss��9=��F�       �	^gI�Yc�A�*

lossx��=,�R:       �	�J�Yc�A�*

loss��=���       �	�ZK�Yc�A�*

loss��=�j��       �	4L�Yc�A�*

loss�/=$�p       �	��L�Yc�A�*

lossW�=e��d       �	h�M�Yc�A�*

loss��=o��       �	��N�Yc�A�*

lossd��<��       �	rjO�Yc�A�*

loss=d=>L(       �	�iP�Yc�A�*

loss9?=☨       �	NQ�Yc�A�*

loss��<|%{�       �	��Q�Yc�A�*

loss���<��\       �	\�R�Yc�A�*

loss	��<
�^       �	�RS�Yc�A�*

loss}�;=@��R       �	�S�Yc�A�*

loss,��<�Q/x       �	@�T�Yc�A�*

loss�=�+6�       �	4HU�Yc�A�*

losse :=�2�       �	��U�Yc�A�*

loss�Q	<�       �	 }V�Yc�A�*

lossh�/<0`�       �	F"W�Yc�A�*

loss�ִ<.E�S       �	�lX�Yc�A�*

loss��5<.�       �	�Y�Yc�A�*

loss���<�S�:       �	�Y�Yc�A�*

loss��"=$��P       �	cZ�Yc�A�*

loss��=��8�       �	�[�Yc�A�*

loss�ݣ<i3�       �	��[�Yc�A�*

loss�J<=���~       �	tD\�Yc�A�*

lossd��=��.�       �	��\�Yc�A�*

loss��W<	��       �	~�]�Yc�A�*

loss���<飅       �	�*^�Yc�A�*

loss.�<C]�       �	K�^�Yc�A�*

loss^D�=���       �	Xs_�Yc�A�*

loss1�=�)       �	#`�Yc�A�*

loss�=\;{       �	D�`�Yc�A�*

loss`�b=^       �	;oa�Yc�A�*

loss���;��ʠ       �	fb�Yc�A�*

lossĐ=�P�       �	f�b�Yc�A�*

loss���<���;       �	O[c�Yc�A�*

loss!��<-��       �	P�c�Yc�A�*

loss^=�u��       �	1�z�Yc�A�*

loss�.!=q%��       �	�i{�Yc�A�*

loss=��=�.�I       �	�|�Yc�A�*

loss�T=�Ӛ�       �	��|�Yc�A�*

lossF�=L��       �	q}�Yc�A�*

loss2�<;}"~       �	9	~�Yc�A�*

lossv��<P��$       �		�~�Yc�A�*

lossO;�<�݌�       �	���Yc�A�*

loss�p�=�ű       �	p$��Yc�A�*

lossUn�=n'e       �	�ʀ�Yc�A�*

loss�b�<#r�       �	�l��Yc�A�*

loss�5�={#       �	E��Yc�A�*

lossF	=�9o�       �	%"��Yc�A�*

lossQ�=�t�       �	���Yc�A�*

lossO�<ٳ�       �	���Yc�A�*

loss@�b=�R�       �	����Yc�A�*

loss%�;R�"�       �	�U��Yc�A�*

loss��<���       �	v���Yc�A�*

loss#��<�">       �	���Yc�A�*

loss�(�=���       �	�߈�Yc�A�*

loss4��<�E�&       �	�|��Yc�A�*

loss4<=6M<�       �	d!��Yc�A�*

loss�(�<<�cb       �	ۿ��Yc�A�*

loss��I=+���       �	#f��Yc�A�*

loss��x=Ύ�n       �	;��Yc�A�*

lossv�<p��       �	����Yc�A�*

loss�"==[       �	}=��Yc�A�*

lossn[^<�.�       �	�ݍ�Yc�A�*

loss�|%=U^5�       �	�y��Yc�A�*

loss`�<k�S       �	���Yc�A�*

loss�,="���       �	����Yc�A�*

loss�1�<~B{       �	~U��Yc�A�*

loss�B=	��       �	;���Yc�A�*

loss���=�+       �	����Yc�A�*

loss��<���       �	L��Yc�A�*

loss[�=�T5       �	���Yc�A�*

loss!�!<��?�       �	BΓ�Yc�A�*

lossSo�=���G       �	{��Yc�A�*

loss�v>HO<       �	���Yc�A�*

loss=�i=�J-�       �	����Yc�A�*

loss�=����       �	f��Yc�A�*

lossz�=&ǉL       �	�	��Yc�A�*

loss��P=D��       �	L���Yc�A�*

loss��'=e���       �	
���Yc�A�*

loss�]=�KҚ       �	� ��Yc�A�*

loss��<A&�       �	����Yc�A�*

loss�ʼ<��       �	O��Yc�A�*

loss���<4�g       �	���Yc�A�*

loss�r}=�V)a       �	o���Yc�A�*

loss�mU;���       �	I���Yc�A�*

loss`��<�0�       �	;��Yc�A�*

loss6^�=c��       �	J���Yc�A�*

loss��<N��f       �	�P��Yc�A�*

lossє�=gyh�       �	���Yc�A�*

loss�1C=uJ=�       �	���Yc�A�*

lossx;݅�}       �	�<��Yc�A�*

lossXz<�a       �	8ڠ�Yc�A�*

loss�g�;d�D,       �	4���Yc�A�*

lossLW�="���       �	���Yc�A�*

loss���=o�5       �	V���Yc�A�*

loss/p=>��       �	P��Yc�A�*

loss��<f��V       �	���Yc�A�*

lossδ�<����       �	P���Yc�A�*

loss:�=$X�       �	{0��Yc�A�*

lossÞ�<oK       �	�ƥ�Yc�A�*

loss�=�
       �	����Yc�A�*

loss�)�=�2��       �	G��Yc�A�*

loss�]�<��\�       �	J���Yc�A�*

lossx��=�Bm       �	aS��Yc�A�*

lossd�>[T��       �	w���Yc�A�*

losst^=�͑�       �	~���Yc�A�*

loss�Yv=o��:       �	s*��Yc�A�*

loss#H�<�ZY       �	�Ī�Yc�A�*

lossw�=�It       �	�\��Yc�A�*

loss۞<:�f�       �	M���Yc�A�*

loss�ą<�x��       �	a���Yc�A�*

loss$��<B�Ǖ       �	�#��Yc�A�*

loss�=y�P�       �	����Yc�A�*

loss���<��e�       �	7p��Yc�A�*

losso�<�lM�       �	]��Yc�A�*

loss��=��       �	4���Yc�A�*

lossS�=:���       �	�W��Yc�A�*

loss�u>R'c2       �	��Yc�A�*

loss�՟<�J�P       �	O���Yc�A�*

lossRK�<w G�       �	�,��Yc�A�*

loss/,G=�m�^       �	�+��Yc�A�*

loss-�#=aQ��       �	Xɳ�Yc�A�*

loss^@=�?i�       �	�w��Yc�A�*

loss���<A�͈       �	U��Yc�A�*

loss��z<oH�       �	Ͽ��Yc�A�*

loss��.= �h�       �	�m��Yc�A�*

loss!*=A��       �	���Yc�A�*

loss��<�G�b       �	靷�Yc�A�*

loss�p�=Ѹ�       �	B��Yc�A�*

loss<0�=	�       �	���Yc�A�*

loss	�< �D{       �	X���Yc�A�*

loss��8=q�NT       �	�"��Yc�A�*

loss�=��q       �	$���Yc�A�*

loss� �=a���       �	�U��Yc�A�*

loss��p=s��*       �	���Yc�A�*

loss�'=<��        �	��Yc�A�*

loss���<"{��       �	���Yc�A�*

loss�'>�V�^       �	ܷ��Yc�A�*

lossxo=��i       �	nO��Yc�A�*

loss,��=���       �	:��Yc�A�*

loss�y�=&��       �	����Yc�A�*

loss(��<��r�       �	&��Yc�A�*

loss�o=O�b�       �	����Yc�A�*

lossv�r=��K�       �	�J��Yc�A�*

loss�=��B       �	����Yc�A�*

loss��>ta        �	����Yc�A�*

loss�6= cZ�       �	�F��Yc�A�*

lossr�C=F>ܭ       �	~���Yc�A�*

loss��<� �       �	����Yc�A�*

loss[�[=m�"       �	Z-��Yc�A�*

loss��<�V�N       �	����Yc�A�*

lossn��<�+��       �	����Yc�A�*

lossq�O=*�\       �	�J��Yc�A�*

loss���<�#p�       �	R���Yc�A�*

lossC� =S��       �	\���Yc�A�*

loss�=����       �	D5��Yc�A�*

loss}�2=G��i       �	����Yc�A�*

lossX�=����       �	�h��Yc�A�*

loss���<�e�r       �	���Yc�A�*

lossڢ�;dTD�       �	s���Yc�A�*

loss;*=ʈ�       �	�;��Yc�A�*

loss�^�=��
�       �	����Yc�A�*

loss��t<w� 8       �	�l��Yc�A�*

lossMc=At�       �	{��Yc�A�*

lossf��=���       �	����Yc�A�*

loss��E=��1       �	RC��Yc�A�*

lossWL�=�%t3       �	���Yc�A�*

loss�AP=�e��       �	(}��Yc�A�*

lossK
=2�        �	� ��Yc�A�*

lossm�=�`�       �	¿��Yc�A�*

loss�R=}�       �	$a��Yc�A�*

loss� =
=�       �	F��Yc�A�*

loss��=���       �	����Yc�A�*

lossX�=�       �	$F��Yc�A�*

loss`�<c�#�       �	&���Yc�A�*

loss���;��       �	���Yc�A�*

loss�ƪ<�@�L       �	���Yc�A�*

loss��L=k@�t       �	����Yc�A�*

lossl3=PW��       �	�Q��Yc�A�*

loss��=�V�       �	����Yc�A�*

loss?=j��       �	I���Yc�A�*

loss��<����       �	jO��Yc�A�*

loss�N><�>��       �	����Yc�A�*

loss�)�<ĸU
       �	Á��Yc�A�*

loss��	=�\o�       �	*��Yc�A�*

loss&�=�KHS       �	����Yc�A�*

lossH!W=T�u#       �	����Yc�A�*

loss!w�=:�       �	�D��Yc�A�*

loss��=�=u       �	����Yc�A�*

lossߜ�=XHu       �	�q��Yc�A�*

loss��<#��0       �	���Yc�A�*

loss$F=ڷ�       �	�#��Yc�A�*

losss�)=5_UN       �	]���Yc�A�*

losst��=7Q        �	�[��Yc�A�*

lossLy
=�]�"       �	7���Yc�A�*

loss%�>]��       �	n���Yc�A�*

loss/�<��޵       �	6��Yc�A�*

loss{i=�x��       �	{���Yc�A�*

loss���;ο�       �	�o��Yc�A�*

loss��Y=Ƒ �       �	��Yc�A�*

losss!=A���       �	����Yc�A�*

lossT�,<P`�       �	����Yc�A�*

loss��=�;��       �	s���Yc�A�*

loss���<>r�       �	�d��Yc�A�*

lossTxf=��       �	���Yc�A�*

loss��_=��?       �	���Yc�A�*

loss�<B��       �	F��Yc�A�*

lossA-=��	c       �	z���Yc�A�*

loss}'P=Y�Pf       �	�v��Yc�A�*

loss��;�x�(       �	���Yc�A�*

loss��=�ok�       �	&���Yc�A�*

loss��D=���       �	%?��Yc�A�*

loss��=M�x       �	'���Yc�A�*

loss�r�<�@
�       �	Lo��Yc�A�*

lossR�=�'n?       �	���Yc�A�*

loss���;�[]�       �	����Yc�A�*

loss���=zVǬ       �	�G��Yc�A�*

loss�3J=2��5       �	����Yc�A�*

loss	M�;<       �	�s��Yc�A�*

loss}Ջ<>I       �	�'��Yc�A�*

loss��=��       �	E���Yc�A�*

loss]]=��k�       �	�[��Yc�A�*

loss83=��n�       �	 ���Yc�A�*

loss�n:=`�{�       �	����Yc�A�*

loss��`=$�ǜ       �	���Yc�A�*

loss��=�?       �	���Yc�A�*

loss�X=Ұ��       �	5]��Yc�A�*

loss$�l<�xh�       �	h��Yc�A�*

loss8p�<+�       �	H���Yc�A�*

loss�p'<WND�       �	����Yc�A�*

loss��=n�B7       �	�"��Yc�A�*

loss�.�<f�N�       �	����Yc�A�*

loss�޵=cLǺ       �	U��Yc�A�*

loss��=W���       �	����Yc�A�*

lossZT�<���        �	P���Yc�A�*

lossz�8=�xx�       �	bL��Yc�A�*

loss�VZ=���       �	����Yc�A�*

loss��D=(#x       �	����Yc�A�*

loss'�
=�b@        �	��Yc�A�*

loss��"<.OHR       �	n���Yc�A�*

lossW=��s4       �	����Yc�A�*

lossVfY=ϊ �       �	x) �Yc�A�*

loss�f=05}Q       �	�� �Yc�A�*

loss{�=n�&       �	v�Yc�A�*

loss�5=�#�c       �	��Yc�A�*

loss&U�<N�{�       �	q��Yc�A�*

lossMa|=��8       �	�c�Yc�A�*

loss��<kI��       �	,�Yc�A�*

loss3�W<4Ÿ:       �	��Yc�A�*

loss [p=~k�       �	�H�Yc�A�*

loss�Q=e��       �	���Yc�A�*

loss��<��ܓ       �	���Yc�A�*

loss��=�p{       �	�9�Yc�A�*

loss]^�=�do�       �	���Yc�A�*

loss���=/"~V       �	���Yc�A�*

lossQ%x<��"�       �	y"	�Yc�A�*

loss;\�<��3<       �	�	�Yc�A�*

lossCJS=�ƭ�       �	��
�Yc�A�*

lossjD�=�|�       �	�+�Yc�A�*

loss��*=�K�       �	���Yc�A�*

losss��<�sz%       �	�p�Yc�A�*

lossxh�=K�7_       �	/�Yc�A�*

loss�AO=���       �	5��Yc�A�*

loss��8=SPn       �	���Yc�A�*

loss��<��B3       �	xE�Yc�A�*

lossCV�<�w��       �	���Yc�A�*

loss�)�;����       �	I��Yc�A�*

lossnG=�qа       �	�)�Yc�A�*

loss*'=�D�0       �	Y��Yc�A�*

loss���=&�~�       �	o��Yc�A�*

loss�`=��       �	��Yc�A�*

loss�=?�|       �	W��Yc�A�*

loss�K=d�"�       �	{N�Yc�A�*

loss@�\=�O|f       �	��Yc�A�*

lossDe<D��/       �	I��Yc�A�*

loss)�=���       �	*�Yc�A�*

loss��H=~'�       �	���Yc�A�*

loss�$�=ӎ��       �	F`�Yc�A�*

loss<أ<!�)       �	���Yc�A�*

loss�D�=��Ze       �	���Yc�A�*

loss� <=5wOL       �	3�Yc�A�*

loss1C�<"��W       �	���Yc�A�*

losshZ�<���       �	��Yc�A�*

lossJ�3=��       �	L�Yc�A�*

lossW(�<�ή�       �	y��Yc�A�*

loss�t=���^       �	�\�Yc�A�*

loss�$=a�       �	�L�Yc�A�*

lossh
=�Nr       �	z��Yc�A�*

loss��<���       �	>v�Yc�A�*

loss�"<쑥2       �	���Yc�A�*

loss;��<k]I       �	�r �Yc�A�*

loss�_�=���       �	�`!�Yc�A�*

loss��Q=��}       �	<"�Yc�A�*

loss��<�w�       �	�"�Yc�A�*

loss��*=q��       �	ǂ#�Yc�A�*

lossH��=��^       �	{�$�Yc�A�*

loss�F=��;=       �	jg%�Yc�A�*

loss�[�<�� 1       �	&�Yc�A�*

loss.'=�>��       �	��&�Yc�A�*

loss���=^�.       �	Ad'�Yc�A�*

lossrя<,�K�       �	x(�Yc�A�*

loss��=��|       �	*�(�Yc�A�*

loss��<2dɚ       �	�K)�Yc�A�*

loss	C=�3�       �	�)�Yc�A�*

loss�ږ=���x       �	��*�Yc�A�*

loss�s_=��r       �	c++�Yc�A�*

loss�$�<����       �	w�+�Yc�A�*

lossL6'=��1F       �	�x,�Yc�A�*

loss�c�==-�       �	�-�Yc�A�*

loss7�=��       �	`�-�Yc�A�*

lossC��=��       �	H.�Yc�A�*

loss�ҽ<�� �       �	*�.�Yc�A�*

loss�;"��s       �	�/�Yc�A�*

loss���=����       �	�0�Yc�A�*

loss\_E=e\�       �	��0�Yc�A�*

loss�=�;�g]       �	��1�Yc�A�*

lossN�<�9n�       �	6#2�Yc�A�*

loss�U�=�L       �	�2�Yc�A�*

loss��<Llc�       �	�V3�Yc�A�*

loss�=��6�       �	$�3�Yc�A�*

loss��W=�߆�       �	)�4�Yc�A�*

loss�]	=�BN�       �	�25�Yc�A�*

loss�#�<����       �	��5�Yc�A�*

loss�?=�[       �	%x6�Yc�A�*

loss��7<�n       �	$7�Yc�A�*

loss�2�;��л       �	/�7�Yc�A�*

loss/�=�z�       �	vT8�Yc�A�*

loss4��;���       �	 �8�Yc�A�*

loss��%<�k��       �	Ѐ9�Yc�A�*

loss�"�<4��       �	:�Yc�A�*

loss��b=�o;       �	ߧ:�Yc�A�*

loss!�=R@�       �	MK;�Yc�A�*

loss�~�=�@�M       �	t�;�Yc�A�*

lossC��=��:�       �	��<�Yc�A�*

loss�^t=����       �	|&=�Yc�A�*

loss�2=o��       �	��=�Yc�A�*

lossW��<}�       �	�{>�Yc�A�*

losswQ=,��       �	�)?�Yc�A�*

loss��=��U�       �	�?�Yc�A�*

lossò=d��       �	�r@�Yc�A�*

loss��$=�Z�}       �	]A�Yc�A�*

loss�,�;̆dT       �	s�B�Yc�A�*

loss�\n=��       �	%�C�Yc�A�*

lossOz�;a�w�       �	AGD�Yc�A�*

loss���=Z#E       �	�E�Yc�A�*

lossE��<cR�Y       �	�)F�Yc�A�*

loss��=Pe�       �	��F�Yc�A�*

loss�;�<ļ��       �	�cG�Yc�A�*

loss�X�<�e��       �	�H�Yc�A�*

loss��=|���       �	��H�Yc�A�*

loss��/=��?�       �	xBI�Yc�A�*

lossJ��=�Vz�       �	f�I�Yc�A�*

loss��h<ܗ��       �	/�J�Yc�A�*

loss���=s��`       �	tBK�Yc�A�*

loss��=0��Z       �	��K�Yc�A�*

loss���<K��       �	��L�Yc�A�*

loss�͜=m�       �	!M�Yc�A�*

loss��=���       �	�M�Yc�A�*

loss}f%=G��       �	�sN�Yc�A�*

loss�uG=f��       �	�
O�Yc�A�*

lossj�Q=�=�       �	��O�Yc�A�*

loss�S =Qo�f       �	eP�Yc�A�*

loss�z�<��~       �	�P�Yc�A�*

loss]�L=(<�=       �	R�Yc�A�*

lossfF!=Q��U       �	��R�Yc�A�*

loss_�<�B!c       �	O�S�Yc�A�*

loss���<��        �	j.T�Yc�A�*

loss�dr<��       �	b�U�Yc�A�*

lossŪ3=�˗       �		qV�Yc�A�*

loss�ؼ=*�J       �	�	W�Yc�A�*

loss��U=Ze%8       �	{�W�Yc�A�*

lossi =G~�       �	�6X�Yc�A�*

loss<^=~�}       �	\�X�Yc�A�*

loss��,=*���       �	�bY�Yc�A�*

lossT�<��-$       �	�Z�Yc�A�*

loss�w=(V�       �	�Z�Yc�A�*

loss�/�=	&��       �	�[�Yc�A�*

loss$�<<(V�       �	Ii\�Yc�A�*

loss�;�<����       �	]�Yc�A�*

loss}��=@�`�       �	��]�Yc�A�*

loss��=�w��       �	�^�Yc�A�*

lossn�t<��#�       �	|�_�Yc�A�*

loss���=�HT�       �	ff`�Yc�A�*

loss�-=����       �	Ga�Yc�A�*

loss�&= ��       �	�a�Yc�A�*

losse	=�JǮ       �	mTb�Yc�A�*

loss�u =T�>       �	�c�Yc�A�*

loss?�<�_1       �	��c�Yc�A�*

loss\�<N/Ҡ       �	�Ld�Yc�A�*

loss�Vt=s��       �	��d�Yc�A�*

loss��<01�       �	Y�e�Yc�A�*

loss���<}��       �	1Df�Yc�A�*

losss�F<D�p       �	�f�Yc�A�*

loss �<�	h�       �	��g�Yc�A�*

loss�H<7(��       �	/h�Yc�A�*

loss��<-��`       �	b�h�Yc�A�*

loss��<�D�s       �	�{i�Yc�A�*

loss�y=1R>�       �	� j�Yc�A�*

loss9׋=�       �	t�j�Yc�A�*

loss���=#�U       �	'�k�Yc�A�*

loss�CM=�-��       �	�+l�Yc�A�*

loss|�<=��=�       �	��l�Yc�A�*

loss �4=���E       �	τm�Yc�A�*

loss�)[<ő}=       �	L6n�Yc�A�*

loss��+=1kp       �	�n�Yc�A�*

loss���<��	       �	Byo�Yc�A�*

loss5A�=3ov�       �	�p�Yc�A�*

loss���=�QU�       �	��p�Yc�A�*

loss�%=�s��       �	�Yq�Yc�A�*

lossC�= q��       �	��q�Yc�A�*

loss�a�<�{�       �	��r�Yc�A�*

loss��N=zZ��       �	�As�Yc�A�*

loss��l=��Q�       �	��s�Yc�A�*

lossX��<3B��       �	��t�Yc�A�*

loss�Vu=a��       �	/�u�Yc�A�*

loss��<6��       �	�-v�Yc�A�*

loss(�=��j(       �	��v�Yc�A�*

loss�1"<{��       �	�qw�Yc�A�*

loss ��<���t       �	fx�Yc�A�*

loss�n><��S       �	{�x�Yc�A�*

loss��<,��(       �	ey�Yc�A�*

loss���=� �z       �	�z�Yc�A�*

loss��<���       �	5�z�Yc�A�*

lossH��<He�       �	�c{�Yc�A�*

losss�q<4�f�       �	,|�Yc�A�*

loss���=��#�       �	��|�Yc�A�*

loss��=�W1       �	�e}�Yc�A�*

loss���=��
�       �	�~�Yc�A�*

lossr�=��i\       �	�~�Yc�A�*

loss&�<����       �	]�Yc�A�*

loss;�g=|�c       �	~��Yc�A�*

loss��<���       �	����Yc�A�*

loss��<HZ�       �	%<��Yc�A�*

loss�w�<$��       �	����Yc�A�*

loss`�=ᤅ       �	���Yc�A�*

loss���;֖L       �	;��Yc�A�*

loss��$=�kp�       �	�؃�Yc�A�*

loss8-=��       �	���Yc�A�*

loss���<�
       �	U��Yc�A�*

lossj=��t       �	����Yc�A�*

lossl�<t`-�       �	����Yc�A�*

lossl;n=���       �	B��Yc�A�*

loss��;=?x�       �	��Yc�A�*

loss�<}��       �	����Yc�A�*

lossTHe=�J��       �	�[��Yc�A�*

losst)�=�/k�       �	���Yc�A�*

loss?2v=d��       �	�9��Yc�A�*

loss)�<*@��       �	ԍ�Yc�A�*

loss�*�=�?N�       �	���Yc�A�*

loss$|=�To�       �	v���Yc�A�*

losshث=F-�       �	�K��Yc�A�*

loss��x;p��       �	���Yc�A�*

loss�̏<�=�       �	@���Yc�A�*

loss��=�       �	�L��Yc�A�*

loss4~�=��Q�       �	���Yc�A�*

loss���<���       �	�Ɠ�Yc�A�*

loss��=[*aE       �	�f��Yc�A�*

loss<VN<M�;       �	���Yc�A�*

loss�g�=����       �	]���Yc�A�*

lossl|�=�3�y       �	�I��Yc�A�*

loss��"=^�;�       �	���Yc�A�*

loss�|=����       �	����Yc�A�*

lossi�l=�>��       �	a3��Yc�A�*

lossڀ=F"�       �	[Ә�Yc�A�*

loss���<+TS�       �	yx��Yc�A�*

loss�sE=J�<3       �	v���Yc�A�*

loss�'=��̔       �	eP��Yc�A�*

loss�=��~       �	9��Yc�A�*

loss��=H�J       �	�Ȝ�Yc�A�*

lossfvO=�)�       �	�n��Yc�A�*

loss�.='��       �	�.��Yc�A�*

loss�W�<�xB�       �	�͞�Yc�A�*

lossʊ�<�'��       �	�g��Yc�A�*

loss&��<yJoY       �	��Yc�A�*

loss�l�<�wc       �	"���Yc�A�*

lossxNd==���       �	sc��Yc�A�*

loss��>�b��       �	a���Yc�A�*

loss2z�<����       �	s���Yc�A�*

loss�;(=��U       �	;7��Yc�A�*

loss���<0b;       �	'ݣ�Yc�A�*

losst�=�Q       �	���Yc�A�*

loss~�=�L�.       �	�%��Yc�A�*

loss�=���       �	*ĥ�Yc�A�*

loss�y=��       �	=`��Yc�A�*

loss&P=��       �	����Yc�A�*

loss�L�=�p�       �	����Yc�A�*

loss�C�<9��       �	b���Yc�A�*

lossJ��<[�C       �	���Yc�A�*

loss�L=�,�       �	?���Yc�A�*

loss�	Y=yF4�       �	�?��Yc�A�*

losss�=H�       �	�֪�Yc�A�*

lossan'=$X��       �	jk��Yc�A�*

loss\��<��       �	Z.��Yc�A�*

loss��<&aR       �	�ˬ�Yc�A�*

loss��=��o       �	}]��Yc�A�*

loss��a<K���       �	����Yc�A�*

loss_<��       �	����Yc�A�*

loss���;2       �	^0��Yc�A�*

loss�Dp=��j       �	�ʯ�Yc�A�*

loss1��=�o\       �	1`��Yc�A�*

loss�=PI       �	L���Yc�A�*

lossv4d<&�>       �	����Yc�A�*

loss�$J<�+�       �	�9��Yc�A�*

lossz6�<[A�       �	ϲ�Yc�A�*

loss�M�<��A�       �	����Yc�A�*

loss�2�<W*X�       �	nN��Yc�A�*

loss3�+=�~�c       �	���Yc�A�*

loss�<�<���J       �	�~��Yc�A�*

loss��Z=��É       �	n��Yc�A�*

loss�s�<���       �	:���Yc�A�*

loss��	<�=*$       �	S[��Yc�A�*

loss���<U�LY       �	���Yc�A�*

loss�IK=���b       �	I���Yc�A�*

loss:��<���5       �	���Yc�A�*

loss�z=���       �	"ǹ�Yc�A�*

loss,��=�0�L       �	h��Yc�A�*

loss�=�.{       �	����Yc�A�*

loss�tV=V49�       �	����Yc�A�*

loss�K�=g�       �	�:��Yc�A�*

lossa��<h�]       �	ּ�Yc�A�*

lossqO�<f7�       �	v��Yc�A�*

loss �D<y>�%       �	0��Yc�A�*

lossr�=9�n[       �	鸾�Yc�A�*

loss�0�<�k       �	GY��Yc�A�*

loss=�i=>��       �	���Yc�A�*

loss�1t='��3       �	~���Yc�A�*

losss�=��N�       �	x)��Yc�A�*

loss�Ŏ;b@�<       �	O���Yc�A�*

loss��=T���       �	�i��Yc�A�*

loss��=����       �	u��Yc�A�*

loss��A<�{��       �	����Yc�A� *

lossH��=�y/`       �	�x��Yc�A� *

loss�K�<8�#�       �	���Yc�A� *

lossEJ�<H�g�       �	���Yc�A� *

loss��s=�:�       �	ܷ��Yc�A� *

loss��.=t#l�       �	8g��Yc�A� *

loss ��<<>�$       �	���Yc�A� *

loss�z�<Ne��       �	F%��Yc�A� *

lossw��<=��u       �	����Yc�A� *

lossIf=��4       �	�q��Yc�A� *

loss��d<޶M9       �	���Yc�A� *

lossֵa=�ٖ�       �	.���Yc�A� *

loss׊�<3J��       �	����Yc�A� *

lossߏ�=Z<�       �	<���Yc�A� *

loss[0
=�kU*       �	) ��Yc�A� *

loss%��<�o�V       �	@��Yc�A� *

loss�˿=Ȉ�`       �	����Yc�A� *

loss�|(<+�ˬ       �	����Yc�A� *

lossiY�=FԲ�       �	H���Yc�A� *

loss �=��!�       �	�A��Yc�A� *

lossd��<�I�F       �	����Yc�A� *

lossͪ=��:       �	����Yc�A� *

loss�U�<o
�J       �	�#��Yc�A� *

loss���<|ꉐ       �	����Yc�A� *

loss��$=�g�       �	w��Yc�A� *

loss#>q=�(�       �	_��Yc�A� *

loss:r=�(       �	����Yc�A� *

loss?@<�](       �	�7��Yc�A� *

loss��K=[|�z       �	����Yc�A� *

lossCR=
Je�       �	��Yc�A� *

loss�h�=�B	       �	[&��Yc�A� *

loss��<0��?       �	����Yc�A� *

loss��+=�
       �	%]��Yc�A� *

losst�w=�=       �	���Yc�A� *

loss�*�<i��       �	���Yc�A� *

loss�s<�ZoM       �	�B��Yc�A� *

loss��k=;<B'       �	T���Yc�A� *

loss:ю<��J       �	���Yc�A� *

lossn��;���[       �	����Yc�A� *

lossiia<Ҽ
�       �	�H��Yc�A� *

lossT.�;�ȤX       �	����Yc�A� *

loss,�;�5C       �	|���Yc�A� *

loss:�<�]
m       �	B���Yc�A� *

loss_C3<�J�S       �	�,��Yc�A� *

loss���<N�+       �	9���Yc�A� *

loss�n<KG�       �	Ox��Yc�A� *

loss_"R:"��       �	���Yc�A� *

loss[�q;����       �	����Yc�A� *

loss(�<=��r       �	��Yc�A� *

loss�z@=����       �	E��Yc�A� *

loss�Y�<���       �	O���Yc�A� *

loss]�; ��V       �	|��Yc�A� *

loss��<����       �	eU��Yc�A� *

loss��>���       �	����Yc�A� *

loss���;m�"       �	����Yc�A� *

loss!��;���       �	�-��Yc�A� *

lossntw=�+��       �		���Yc�A� *

loss� >��1Y       �	�]��Yc�A� *

loss�{�<��"       �	0���Yc�A� *

loss�R_<NV��       �	���Yc�A� *

loss�(�=e΢9       �	�5��Yc�A� *

loss�m�=��<       �	5���Yc�A� *

loss�"J=���       �	�j��Yc�A� *

loss��=%}��       �	r���Yc�A� *

loss}��<��e       �	
���Yc�A� *

loss��<��m       �	�:��Yc�A� *

loss<�0=���w       �	=���Yc�A� *

loss��<YO��       �	b���Yc�A� *

lossH~�=չܴ       �	�#��Yc�A� *

loss$�H=�[_�       �	Ӿ��Yc�A� *

loss��I=����       �	�^��Yc�A� *

lossv�=��-D       �	����Yc�A� *

loss7.�<'��       �	ɏ��Yc�A� *

loss��<���Z       �	�-��Yc�A� *

loss��;f�r�       �	����Yc�A� *

loss��<��?�       �	�^��Yc�A� *

lossns<�_�7       �	X��Yc�A� *

loss�<*<ġ��       �	|���Yc�A� *

lossl;�<�z       �	{/��Yc�A� *

loss.<0vج       �	����Yc�A� *

lossw�<@l�|       �	Bz��Yc�A� *

lossjo�<\��       �	�>��Yc�A� *

loss���<Ƥ�       �	(���Yc�A� *

lossC=���       �	zp��Yc�A� *

loss<�i�p       �	k��Yc�A� *

loss�2�=v�)�       �	����Yc�A� *

loss
��;���       �	���Yc�A� *

loss�5&<�n       �	�A �Yc�A� *

loss�K"=�z�W       �	]� �Yc�A� *

losso�<��?2       �	;r�Yc�A� *

loss""=�>       �	��Yc�A� *

loss�c�<���       �	I��Yc�A� *

loss��=+y�       �	�u�Yc�A� *

loss���=��:       �	9�Yc�A� *

loss?�'=���       �	/��Yc�A� *

loss�Q =�lT       �	C�Yc�A� *

loss�=`�d�       �	���Yc�A� *

loss���=����       �	Ӄ�Yc�A� *

loss�7i<�oCi       �	N^�Yc�A� *

loss��4=�T��       �	r��Yc�A� *

loss=!s<��x       �	���Yc�A� *

lossj <<��g�       �	oe	�Yc�A� *

loss�(=��y�       �	H�	�Yc�A� *

loss��s<�w��       �	�
�Yc�A� *

loss�E�<���       �	�.�Yc�A� *

loss�U<����       �	px!�Yc�A� *

lossq}�=_���       �	�"�Yc�A� *

loss���=����       �	s�"�Yc�A� *

lossy=�X�J       �	��#�Yc�A� *

lossh;g=�r�       �	�$�Yc�A� *

loss
w�<bm�a       �	��$�Yc�A� *

lossP��=�q       �	~R%�Yc�A� *

loss��v=���Q       �	u�%�Yc�A� *

loss�Z�=$�"       �	��&�Yc�A� *

loss��K=�s�;       �	�V'�Yc�A� *

loss�ށ<�I;       �	�'�Yc�A� *

loss���<�V$       �	u�(�Yc�A� *

loss�k*=&��R       �	�()�Yc�A� *

lossC�=0w$�       �	��)�Yc�A� *

lossK�=�N       �	dW*�Yc�A� *

loss3�~=��f       �	�*�Yc�A� *

loss��;ٷ�       �	|-�Yc�A� *

loss��=����       �	j�-�Yc�A� *

loss��=)�$�       �	4G.�Yc�A� *

loss���=7��4       �	��.�Yc�A� *

loss�JZ=�+EX       �	>v/�Yc�A� *

lossA��<P ��       �	�0�Yc�A� *

loss9�<��D�       �	�0�Yc�A� *

lossnV=O���       �	�U1�Yc�A�!*

loss�Y=!yD       �	�:2�Yc�A�!*

loss-�<vJ�#       �	��2�Yc�A�!*

loss�N�<�l��       �	��3�Yc�A�!*

lossK�;�W�       �	[%4�Yc�A�!*

loss
;�<+�F       �	�4�Yc�A�!*

loss��=@q       �	�`5�Yc�A�!*

lossN�<�3�F       �	�6�Yc�A�!*

loss�D=eڒ�       �	�6�Yc�A�!*

loss��!=P��{       �	jL7�Yc�A�!*

lossy�=�$�       �	��7�Yc�A�!*

loss��=�eҀ       �	��8�Yc�A�!*

loss�}y=�h�       �	$E9�Yc�A�!*

loss�L�<�mdc       �	(�9�Yc�A�!*

lossl�=�V"�       �	��:�Yc�A�!*

lossv�>b<��       �	�";�Yc�A�!*

loss�n=��?e       �	��;�Yc�A�!*

loss�+=(>N       �	�Y<�Yc�A�!*

loss3Cz=0Ʋ$       �	�=�Yc�A�!*

loss��7<X�O�       �	�=�Yc�A�!*

loss@��<]+�	       �	�6>�Yc�A�!*

loss�f=-�T       �	��>�Yc�A�!*

loss��=�<��       �	
�?�Yc�A�!*

loss�>$=�ù�       �	@�Yc�A�!*

loss��=A�7       �	��@�Yc�A�!*

lossd@"=���       �	;nA�Yc�A�!*

lossS"<�⸗       �	�B�Yc�A�!*

loss�=��k�       �	s�B�Yc�A�!*

lossN5u=��       �	2UC�Yc�A�!*

loss�Z<�@�       �	R�C�Yc�A�!*

lossܒ>,K�       �	E�D�Yc�A�!*

lossU"="��       �	��E�Yc�A�!*

loss��<��b<       �	 aF�Yc�A�!*

lossJ��;<]�       �	t	G�Yc�A�!*

loss 9Y;&���       �	��G�Yc�A�!*

loss�#=|v�\       �	S�H�Yc�A�!*

loss��=��       �	��I�Yc�A�!*

loss
Ы=BH��       �	U0J�Yc�A�!*

loss#��<�1��       �	!K�Yc�A�!*

lossJ�;~&\�       �	��L�Yc�A�!*

loss���<Vo��       �	�oM�Yc�A�!*

loss?�Q<�:R       �	�9N�Yc�A�!*

loss�u�;=��       �	MIO�Yc�A�!*

loss�d�<MN|)       �		�P�Yc�A�!*

loss���=(�S       �	�,Q�Yc�A�!*

loss�׶=H���       �	)�R�Yc�A�!*

loss���=�|�e       �	XqS�Yc�A�!*

lossH�&=8��       �	T�Yc�A�!*

loss�h�<V�L       �	+�T�Yc�A�!*

loss��=8Y�       �	{gU�Yc�A�!*

lossr�=�F=�       �	GV�Yc�A�!*

losse�G=T^k       �	�V�Yc�A�!*

loss�4=�Q��       �	�HW�Yc�A�!*

loss�b�<xA       �	��W�Yc�A�!*

loss�D�<�yGq       �	��X�Yc�A�!*

lossp <n�9�       �	�Y�Yc�A�!*

lossw��<*m��       �	 Z�Yc�A�!*

lossn�,=�T,       �	�Z�Yc�A�!*

loss���<���J       �	B|[�Yc�A�!*

loss�=��g�       �	�&\�Yc�A�!*

lossW��;�.       �	��\�Yc�A�!*

loss]ݱ<V?�       �	`]�Yc�A�!*

loss��/=�Z��       �	`^�Yc�A�!*

loss,�=�HN       �	{�^�Yc�A�!*

loss��=�6�       �	�C_�Yc�A�!*

loss��<��w       �	'�_�Yc�A�!*

lossL!Q<V�7       �	�y`�Yc�A�!*

loss<�=��۶       �	�a�Yc�A�!*

lossy/#=2�C�       �	�a�Yc�A�!*

loss߮<v]       �	Tpb�Yc�A�!*

lossA�=��Tq       �	O"c�Yc�A�!*

lossK��=���F       �	+�c�Yc�A�!*

loss�7�;)�}�       �	�id�Yc�A�!*

loss8��<�t�       �	�e�Yc�A�!*

loss�4�<%�\       �	 �e�Yc�A�!*

loss��L=ܻ��       �	�Nf�Yc�A�!*

loss��<n��       �	a�f�Yc�A�!*

loss��;~̻       �	#�g�Yc�A�!*

loss(��<���%       �	�oh�Yc�A�!*

loss��=p�s�       �	�i�Yc�A�!*

loss���<���n       �	)�i�Yc�A�!*

loss�}<=���       �	~Uj�Yc�A�!*

loss:Y==��       �	8�j�Yc�A�!*

loss�<�-!       �	k�k�Yc�A�!*

loss�6l<�HZ[       �	�El�Yc�A�!*

loss6��=����       �	��l�Yc�A�!*

lossIȝ<��$�       �	s�m�Yc�A�!*

loss�ii=���       �	�n�Yc�A�!*

loss{f�<����       �	�n�Yc�A�!*

loss���=��IT       �	ȵp�Yc�A�!*

loss�)�;Ol@       �	y]q�Yc�A�!*

loss�Z=�<�*       �	r�Yc�A�!*

loss���=� ��       �	^�r�Yc�A�!*

loss�c=lD7�       �	�fs�Yc�A�!*

loss3��=�Ӌ�       �	�
t�Yc�A�!*

loss��/=j�N�       �	v�t�Yc�A�!*

loss&y=`b\�       �	�Pu�Yc�A�!*

lossG�=�fI       �	�u�Yc�A�!*

lossLG�<��       �	�v�Yc�A�!*

loss���<۴�U       �	obw�Yc�A�!*

loss���<�"��       �	�x�Yc�A�!*

loss�w�<Y�       �	D�x�Yc�A�!*

loss�S=�]Y�       �	Dy�Yc�A�!*

lossh�<^((�       �	��y�Yc�A�!*

loss��X=�M��       �	��z�Yc�A�!*

loss ��=)<�a       �	�*{�Yc�A�!*

loss�u=O !       �	)�{�Yc�A�!*

loss��0=�k�|       �	�h|�Yc�A�!*

loss��G=|�/:       �	31}�Yc�A�!*

lossHM�<�ky�       �	q�}�Yc�A�!*

loss�<=��*h       �	�e~�Yc�A�!*

loss �L=�D��       �	��Yc�A�!*

lossNW�<ڽy       �	���Yc�A�!*

loss��K=����       �	�:��Yc�A�!*

loss�Db=���       �	���Yc�A�!*

loss��<��       �	�|��Yc�A�!*

loss�c�<g��       �	���Yc�A�!*

loss��<�`�       �	?���Yc�A�!*

lossïA=�2�z       �	=F��Yc�A�!*

loss�r=��Q�       �	����Yc�A�!*

lossxdN=�.JO       �	�y��Yc�A�!*

loss��=%[0       �	$��Yc�A�!*

loss4��<js@T       �	����Yc�A�!*

loss�V�<�I       �	�]��Yc�A�!*

loss�<)�-"       �		��Yc�A�!*

loss�f�<Mޓ�       �	����Yc�A�!*

loss�L!=q�v       �	�M��Yc�A�!*

loss�C=ר��       �	��Yc�A�!*

loss�VI=%
��       �	؂��Yc�A�"*

loss��;=��       �	���Yc�A�"*

loss#9=v6`�       �	�V��Yc�A�"*

loss�8=�P       �	���Yc�A�"*

loss�<�j>3       �	-���Yc�A�"*

loss�g�<�g�       �	�I��Yc�A�"*

lossjv5=KOh�       �	K��Yc�A�"*

loss�_}=��       �	����Yc�A�"*

lossF�=��       �	�"��Yc�A�"*

loss@2�<��A�       �	����Yc�A�"*

loss��<,       �	�^��Yc�A�"*

lossL��<i�fP       �	L���Yc�A�"*

loss@:<p��       �	1���Yc�A�"*

loss2�R=~�?y       �	\���Yc�A�"*

loss)��<Po�S       �	�G��Yc�A�"*

lossq7]<ڰ
#       �	���Yc�A�"*

loss��=y&�$       �	�z��Yc�A�"*

loss�t�<����       �	�V��Yc�A�"*

loss��Z<�X�Y       �	W��Yc�A�"*

loss�?=�7�       �	���Yc�A�"*

loss��g<�M��       �	9���Yc�A�"*

loss!H+<u�U       �	 7��Yc�A�"*

loss�L�=y��       �	[Ҙ�Yc�A�"*

lossqb�<�4�0       �	�w��Yc�A�"*

loss ��<�)�       �	r��Yc�A�"*

loss���=��       �	⯚�Yc�A�"*

loss�h�=?Г�       �	
M��Yc�A�"*

loss$|@=���'       �	q��Yc�A�"*

losse-�;hʝ%       �	�֝�Yc�A�"*

loss�<�*�       �	�q��Yc�A�"*

loss�K=�v
V       �	���Yc�A�"*

loss��<f�q       �	*���Yc�A�"*

lossl�P<V���       �	FC��Yc�A�"*

loss|�<k,��       �	�ߠ�Yc�A�"*

loss��=�]ب       �	�v��Yc�A�"*

lossx�=�b�I       �	D��Yc�A�"*

loss��=.}�#       �	����Yc�A�"*

loss��?=X��       �	m���Yc�A�"*

lossԆ}=�Ř       �	�E��Yc�A�"*

losssr�< A�       �	@ݤ�Yc�A�"*

loss
?�<!�       �	=|��Yc�A�"*

loss���<��N~       �	���Yc�A�"*

loss�ζ<�5�       �	����Yc�A�"*

loss�C2<���       �	DR��Yc�A�"*

lossl=���       �	���Yc�A�"*

loss���<�K��       �	.���Yc�A�"*

lossx��<x3��       �	tA��Yc�A�"*

loss۳<\���       �	�ߩ�Yc�A�"*

loss�.=�Kn       �	A���Yc�A�"*

loss3)/=�E��       �	���Yc�A�"*

lossZ�.=$�;       �	4���Yc�A�"*

loss =���       �	9^��Yc�A�"*

loss*|	=@�2�       �	����Yc�A�"*

loss�i5<��R       �	f���Yc�A�"*

loss��h=��٭       �	?;��Yc�A�"*

loss�t;=�p��       �	� ��Yc�A�"*

loss���<���       �	Ǡ��Yc�A�"*

loss��=��Z�       �	C��Yc�A�"*

loss���<;�DZ       �	�ڰ�Yc�A�"*

loss�!�<���#       �	�}��Yc�A�"*

loss]f={�       �	G ��Yc�A�"*

loss��(<ϩ�A       �	$Ѳ�Yc�A�"*

loss�<%b�       �	hz��Yc�A�"*

loss$�=9��D       �	~��Yc�A�"*

loss�"I=���#       �	1д�Yc�A�"*

loss�@�<W��       �	c��Yc�A�"*

lossSN>�ᇡ       �	�!��Yc�A�"*

lossf9�=��.�       �	�ζ�Yc�A�"*

loss`;=S�#       �	ux��Yc�A�"*

loss��7<R2�       �	���Yc�A�"*

loss%E==�u       �	�ø�Yc�A�"*

loss^(�<)��b       �	8j��Yc�A�"*

loss.��=���       �	���Yc�A�"*

loss�&�=/3�       �	����Yc�A�"*

loss\=�OK�       �	YM��Yc�A�"*

loss�Fi=/��       �	t��Yc�A�"*

lossHC�=�>ұ       �	����Yc�A�"*

loss�M<�       �	�!��Yc�A�"*

lossݵ�<sd�       �	����Yc�A�"*

loss (f<9L��       �	�\��Yc�A�"*

loss��<����       �	;���Yc�A�"*

loss3��=!_��       �	���Yc�A�"*

loss���<�w�e       �	@2��Yc�A�"*

loss���=j?:"       �	)���Yc�A�"*

loss�W�=��       �	�m��Yc�A�"*

loss��G=�
|       �	���Yc�A�"*

loss�D�=ּ��       �	-���Yc�A�"*

loss��7=V�%       �	O��Yc�A�"*

loss���<^5��       �	���Yc�A�"*

loss�Ϯ<}·#       �	����Yc�A�"*

loss��=��ћ       �	���Yc�A�"*

lossW�%=1���       �	���Yc�A�"*

lossJ�_=dG?�       �	����Yc�A�"*

loss��U=�ɽ�       �	�.��Yc�A�"*

loss��<W_{�       �	-��Yc�A�"*

lossl�<����       �	EI��Yc�A�"*

loss��<�?�       �	&���Yc�A�"*

loss��3=�B��       �	֧��Yc�A�"*

loss��=p�U�       �	eS��Yc�A�"*

loss� �=��#0       �	����Yc�A�"*

loss3�<=�}�N       �	l���Yc�A�"*

loss�{=����       �	�2��Yc�A�"*

loss�ݞ<x���       �	����Yc�A�"*

loss��h<Dg��       �	���Yc�A�"*

loss�E�<��'�       �	^*��Yc�A�"*

loss��B=�Q��       �	����Yc�A�"*

loss�~�<��ު       �	���Yc�A�"*

loss5x�<�b9�       �	�&��Yc�A�"*

loss�D�=��wr       �	���Yc�A�"*

loss�3u=ȳe�       �	����Yc�A�"*

loss��=��       �	�U��Yc�A�"*

loss7a�=PWWb       �	����Yc�A�"*

loss?�8=��s       �	%���Yc�A�"*

loss?�g=&��       �	�8��Yc�A�"*

loss`�<2:�       �	���Yc�A�"*

lossO�=���       �	�n��Yc�A�"*

loss�<�Q       �	9
��Yc�A�"*

loss�ߠ=�6��       �	ϡ��Yc�A�"*

loss��a=�%��       �	�<��Yc�A�"*

loss�-�=����       �	9���Yc�A�"*

losso�=����       �	���Yc�A�"*

lossXM=�"��       �	�9��Yc�A�"*

losszc�=گl       �	*���Yc�A�"*

loss{ί=�tZ       �	����Yc�A�"*

loss���=���%       �	L6��Yc�A�"*

loss���=#�}�       �	���Yc�A�"*

lossd�<�g&       �	�{��Yc�A�"*

loss��m=��       �	h ��Yc�A�"*

loss�s"=�'�        �	>���Yc�A�#*

loss�?;��P<       �	�g��Yc�A�#*

loss���<%�U       �	l��Yc�A�#*

lossy�<C�]       �	���Yc�A�#*

loss)�<��db       �	�A��Yc�A�#*

loss���=[�T�       �	����Yc�A�#*

losst!=<�3       �	k���Yc�A�#*

loss��o=.��       �	�X��Yc�A�#*

lossn3%=1d�
       �	E���Yc�A�#*

loss Y<���
       �	����Yc�A�#*

losst��;�z&       �	�-��Yc�A�#*

loss�ʯ;-;f       �	���Yc�A�#*

lossC��<�VX       �	
d��Yc�A�#*

loss��<�1�       �	0��Yc�A�#*

loss�de<&Y��       �	����Yc�A�#*

loss�=^�Ϲ       �	є��Yc�A�#*

loss�7]=�Xs       �	�J��Yc�A�#*

loss�nx<�oڦ       �	����Yc�A�#*

loss�Q(=M?�       �	����Yc�A�#*

loss�r�=��=�       �	�M��Yc�A�#*

losszf�=��j�       �	����Yc�A�#*

loss��<�]��       �	x���Yc�A�#*

loss2��<�WI       �	C8��Yc�A�#*

lossb = ��g       �	���Yc�A�#*

loss4`�=��       �	g���Yc�A�#*

loss���<l��       �	�Y��Yc�A�#*

loss�P=�,r       �	����Yc�A�#*

losst�<��$�       �	���Yc�A�#*

lossN;N=�|�       �	80��Yc�A�#*

loss�*=����       �	����Yc�A�#*

loss�<����       �	�x��Yc�A�#*

loss�=4��       �	��Yc�A�#*

loss� �=���       �	0���Yc�A�#*

loss3=��7�       �	Zb��Yc�A�#*

lossi͂<q6��       �	z ��Yc�A�#*

loss\��<�ܱ�       �	2���Yc�A�#*

loss�i=�Q˦       �	�R��Yc�A�#*

lossEB9=�?e       �	����Yc�A�#*

loss]��=]��       �	U���Yc�A�#*

loss{(J=ӂ�{       �	�$��Yc�A�#*

loss���<��>       �	G���Yc�A�#*

losss�<c38r       �	�z��Yc�A�#*

loss�
u=���D       �	�%��Yc�A�#*

lossE'�=�S$�       �	����Yc�A�#*

lossۿ�;�u�       �	ʇ��Yc�A�#*

loss���<�0�       �	�/��Yc�A�#*

loss��^=�zm�       �	����Yc�A�#*

lossSU�=E�'�       �	,a��Yc�A�#*

loss{�<J��       �	����Yc�A�#*

losshY/=d�       �	`���Yc�A�#*

loss�k$=h�x       �	k( �Yc�A�#*

loss�<v��       �	�� �Yc�A�#*

loss��<�q:       �	^�Yc�A�#*

lossbP�<��'�       �	f��Yc�A�#*

lossj^=���A       �	G��Yc�A�#*

loss�q=8�ET       �	�0�Yc�A�#*

lossźO<FM��       �	��Yc�A�#*

losswP=/���       �	�m�Yc�A�#*

loss1n=Jb��       �	�E�Yc�A�#*

lossXy9=5��       �	���Yc�A�#*

lossmK%<��|�       �	���Yc�A�#*

loss�t�;$��X       �	cb�Yc�A�#*

loss4&3=%�pG       �	�K�Yc�A�#*

loss�h�<K��	       �	k-	�Yc�A�#*

loss�ܐ<ո>�       �	u�	�Yc�A�#*

loss���=�� R       �	h�
�Yc�A�#*

losst(�=y���       �	�|�Yc�A�#*

losss��<��]�       �	�Yc�A�#*

lossO�V=�,S$       �	��Yc�A�#*

loss\�<8�m       �	��Yc�A�#*

loss�u�<�`       �	�R�Yc�A�#*

loss� <`���       �	���Yc�A�#*

loss�<Y!��       �	���Yc�A�#*

lossmc�;\�8       �	� �Yc�A�#*

loss͘<҃�E       �	b��Yc�A�#*

loss�52=��V�       �	�U�Yc�A�#*

lossV�=�G��       �	�q�Yc�A�#*

lossb`�<�~�       �	�
�Yc�A�#*

lossr3x=�!Y?       �	���Yc�A�#*

loss��<�Y�       �	�4�Yc�A�#*

loss3��<�]|       �	��Yc�A�#*

loss��=�F%�       �	4h�Yc�A�#*

loss���<{) �       �	5$�Yc�A�#*

loss��D=�aZ       �	'��Yc�A�#*

lossU��=�q��       �	yY�Yc�A�#*

loss��=����       �	g��Yc�A�#*

losscD*=��v       �	ԙ�Yc�A�#*

loss��=r�MV       �	U2�Yc�A�#*

lossA��<��72       �	X��Yc�A�#*

lossN;�<�[:       �	g_�Yc�A�#*

loss���<C���       �	���Yc�A�#*

loss67�<��*?       �	���Yc�A�#*

loss�p�;��_       �	a3�Yc�A�#*

loss�2J=$E       �	>��Yc�A�#*

loss��<���       �	
e�Yc�A�#*

loss���<���       �	�Yc�A�#*

loss��=G
K�       �	���Yc�A�#*

loss��R<����       �	�i�Yc�A�#*

lossh��<�?�       �	� �Yc�A�#*

loss��<17�       �	�� �Yc�A�#*

loss`��='�FZ       �	�J!�Yc�A�#*

lossd�<���       �	��!�Yc�A�#*

loss�c�<�W�c       �	h�"�Yc�A�#*

loss�pt<�M�"       �	�*#�Yc�A�#*

lossv�	<��8       �	�#�Yc�A�#*

loss)��;��r,       �	�`$�Yc�A�#*

lossV�;�i(       �	6%�Yc�A�#*

loss��2=��V�       �	�%�Yc�A�#*

loss��}=Ъ,$       �	�E&�Yc�A�#*

loss�ˉ<-�3@       �	��&�Yc�A�#*

lossj�y<���       �	=�'�Yc�A�#*

loss�6%=eH�       �	6Z(�Yc�A�#*

loss�ǌ<�-ڏ       �	��(�Yc�A�#*

loss��*=�'�       �	3�)�Yc�A�#*

lossV63=t��       �	�&*�Yc�A�#*

lossMs�<��Q�       �	U�*�Yc�A�#*

loss]�a=�j͏       �	�b+�Yc�A�#*

lossW�;�|>o       �	��+�Yc�A�#*

loss���<b���       �	��,�Yc�A�#*

loss�"�<N�L       �	E-�Yc�A�#*

loss��<����       �	D�-�Yc�A�#*

loss��<����       �	9|.�Yc�A�#*

loss'{=1�       �	!/�Yc�A�#*

loss (�<��       �	�/�Yc�A�#*

loss�<aQ�}       �	�Z0�Yc�A�#*

lossxX�<�,�       �	��0�Yc�A�#*

losstV=f       �	��1�Yc�A�#*

lossxU'=i	��       �	��2�Yc�A�#*

loss�F�=jC�       �	8.3�Yc�A�$*

lossC��;M       �	�3�Yc�A�$*

loss��L=U�}�       �	�l4�Yc�A�$*

loss\�=�Ƌ�       �	t5�Yc�A�$*

lossc�r=����       �	��5�Yc�A�$*

lossa&=�LW�       �	L46�Yc�A�$*

loss��=���       �	�6�Yc�A�$*

loss�/=AQx:       �	��7�Yc�A�$*

loss`��<����       �	�T8�Yc�A�$*

loss}�?;RWh�       �	��8�Yc�A�$*

lossW6�<�X�]       �	��9�Yc�A�$*

loss6�=�ʃ�       �	�:�Yc�A�$*

loss`�=)3�Z       �	��:�Yc�A�$*

loss7��<؉�\       �	(F;�Yc�A�$*

loss���<��V       �	3�;�Yc�A�$*

loss�˪=rJ�z       �	��<�Yc�A�$*

loss���=:|B       �	�T=�Yc�A�$*

loss� =��P�       �	��=�Yc�A�$*

loss��=G���       �	�>�Yc�A�$*

loss�&k=��       �	�)?�Yc�A�$*

loss�=��n�       �	�?�Yc�A�$*

lossȡ�<:�s�       �	�i@�Yc�A�$*

loss��@=_�2       �	5A�Yc�A�$*

loss=��"�       �	v�A�Yc�A�$*

loss�ܚ<�C�2       �	gbB�Yc�A�$*

loss�<�<���       �	~�B�Yc�A�$*

lossv=��Q       �	�C�Yc�A�$*

loss��=3m��       �	_$D�Yc�A�$*

loss���<�^1�       �	v�D�Yc�A�$*

loss��
=�=�K       �	�oE�Yc�A�$*

loss �<�x�       �	�6F�Yc�A�$*

loss)�z<�Q�z       �	��F�Yc�A�$*

loss�IK<l3�       �	��G�Yc�A�$*

loss�yh=}c       �	#jH�Yc�A�$*

loss80�=��^       �	ǃI�Yc�A�$*

loss݉1=��g       �	�J�Yc�A�$*

loss�0=�%�D       �	��J�Yc�A�$*

loss8%�;Ⱦ^       �	�K�Yc�A�$*

loss /=��wd       �	�eL�Yc�A�$*

loss�=�(��       �	a6M�Yc�A�$*

lossG=���i       �	�M�Yc�A�$*

lossCֱ<�4}p       �	�oN�Yc�A�$*

loss�:�<7��       �	 O�Yc�A�$*

loss��=[�~       �	(�O�Yc�A�$*

lossX=�oz|       �	`YP�Yc�A�$*

lossR�<����       �	��P�Yc�A�$*

loss�W=�g[�       �	��Q�Yc�A�$*

lossE�w=�a��       �	+R�Yc�A�$*

losst��=	Y��       �	IS�Yc�A�$*

loss��z=!�(       �	�S�Yc�A�$*

loss�:�<m6�M       �	KT�Yc�A�$*

loss���<F���       �	��T�Yc�A�$*

lossm��<j�q�       �	t�U�Yc�A�$*

lossSZ<��Jv       �	��V�Yc�A�$*

loss�1�<4�V�       �	#W�Yc�A�$*

lossW<J��A       �	<�W�Yc�A�$*

loss1�E=��O       �	��X�Yc�A�$*

loss�~�=(F��       �		Z�Yc�A�$*

loss\}=`��`       �	H�Z�Yc�A�$*

loss�P=^�b�       �	�F[�Yc�A�$*

loss1��;��r�       �	��[�Yc�A�$*

loss�S=�)�m       �	�~\�Yc�A�$*

lossA�<"��1       �	�]�Yc�A�$*

loss��=(Qq       �	�]�Yc�A�$*

loss�$=��&       �	�^�Yc�A�$*

loss<�=��O       �	�._�Yc�A�$*

loss4��=B�|�       �	�_�Yc�A�$*

loss��<�T��       �	�r`�Yc�A�$*

lossu9<��KW       �	5
a�Yc�A�$*

loss��<�b��       �	{�a�Yc�A�$*

loss�O=@�#       �	;b�Yc�A�$*

loss���<�n       �	��b�Yc�A�$*

loss��=�0�       �	4ic�Yc�A�$*

loss��T=�e��       �	Md�Yc�A�$*

lossv�=���d       �	�d�Yc�A�$*

loss�=���       �	�\e�Yc�A�$*

loss�hx=C��7       �	�f�Yc�A�$*

lossA�<߰L�       �	�f�Yc�A�$*

loss�p�<���       �	4g�Yc�A�$*

loss��g<�jU�       �	/�g�Yc�A�$*

loss��=��5�       �	E�h�Yc�A�$*

lossiv<2�Nu       �	�)i�Yc�A�$*

losst��<!��       �	��i�Yc�A�$*

loss���<m3�H       �	Faj�Yc�A�$*

loss&3�=�n�       �	k�Yc�A�$*

lossnd;���       �	��k�Yc�A�$*

lossn��<�6�
       �	�Cl�Yc�A�$*

lossc0�<���       �	Tm�Yc�A�$*

loss<��<B"F�       �	�m�Yc�A�$*

loss(A�=W��       �	�In�Yc�A�$*

loss�"=B��       �	�n�Yc�A�$*

loss�!=�-�b       �	��o�Yc�A�$*

loss�=5`�       �	H2p�Yc�A�$*

loss�v�<�+��       �	O�p�Yc�A�$*

loss���;���2       �	Aeq�Yc�A�$*

lossxٽ<�s�{       �	]�q�Yc�A�$*

loss�=<�u�}       �	S�r�Yc�A�$*

loss�'=C��       �	0s�Yc�A�$*

lossT(�<o�Z�       �	?�s�Yc�A�$*

loss��<�i�/       �	Yt�Yc�A�$*

loss�=@�B�       �	��t�Yc�A�$*

loss�B=B�       �	v�u�Yc�A�$*

loss,�=9QI�       �	g&v�Yc�A�$*

loss�z�<ԡ�N       �	��v�Yc�A�$*

loss{�6=���       �	qw�Yc�A�$*

loss$�< �k       �	�x�Yc�A�$*

loss)�8=���       �	5�x�Yc�A�$*

loss)�<��H�       �	�Ly�Yc�A�$*

loss�ۀ<��y~       �	�y�Yc�A�$*

losso1=ŅD$       �	q�z�Yc�A�$*

lossҤ<��       �	�f{�Yc�A�$*

loss���;%Ff�       �	�|�Yc�A�$*

loss!� =[!�       �	�|�Yc�A�$*

lossD�=lu�X       �	z7}�Yc�A�$*

lossǺ=
��       �	$�}�Yc�A�$*

loss{�<no�5       �	ԛ~�Yc�A�$*

loss�=�ȷ^       �	�7�Yc�A�$*

loss]H=;	9)       �	��Yc�A�$*

loss�1Z="�       �	�s��Yc�A�$*

lossb�=Ҝ�;       �	���Yc�A�$*

loss���<�L1G       �	2���Yc�A�$*

lossݫg=LdM�       �	�G��Yc�A�$*

loss�m<k���       �	L���Yc�A�$*

lossq��<.��       �	�u��Yc�A�$*

loss���<f�^�       �	[	��Yc�A�$*

loss��9;���       �	ס��Yc�A�$*

lossp@�;v���       �	;7��Yc�A�$*

lossv�;)fa       �	!��Yc�A�$*

loss3�<�z"�       �	���Yc�A�%*

loss�)�;�s�       �	3���Yc�A�%*

lossU��<�ܕV       �	�$��Yc�A�%*

loss�<���j       �	�Έ�Yc�A�%*

loss���<�ɼ       �	�p��Yc�A�%*

lossdX�<xSt       �	���Yc�A�%*

loss�;��A       �	���Yc�A�%*

loss��5;3ɳU       �	�A��Yc�A�%*

loss;q�;�|��       �	���Yc�A�%*

loss�K=���$       �	����Yc�A�%*

loss�<<��Y�       �	S��Yc�A�%*

loss���:�?|       �	:��Yc�A�%*

loss��3<ݭ�       �	����Yc�A�%*

loss�,>��Y       �	=��Yc�A�%*

lossQ!C;.�0(       �	��Yc�A�%*

loss2P�;16.       �	됐�Yc�A�%*

loss�-�=_y7N       �	�1��Yc�A�%*

lossV�<OE�V       �	�ʑ�Yc�A�%*

lossL�<�&��       �	�b��Yc�A�%*

lossHmP<�~g�       �	["��Yc�A�%*

loss�p�={_*       �	V���Yc�A�%*

loss�Ҋ=�oj�       �	�V��Yc�A�%*

loss��<��8�       �	R��Yc�A�%*

loss�|)={�x       �	ʈ��Yc�A�%*

loss���=@��       �	���Yc�A�%*

lossaл=4�,�       �	����Yc�A�%*

lossƼf=��(�       �	DR��Yc�A�%*

loss��M=��7       �	���Yc�A�%*

loss� Y=;A�j       �	
���Yc�A�%*

loss�,�=��\       �	���Yc�A�%*

loss\:#=.�m       �	����Yc�A�%*

loss�R<n-�       �	�M��Yc�A�%*

lossH[�=irQ�       �	��Yc�A�%*

loss@;�<���       �	���Yc�A�%*

loss��<	��       �	���Yc�A�%*

loss-3=ý9,       �		4��Yc�A�%*

lossj-<熾�       �	���Yc�A�%*

loss�D�<�ϳ�       �	y��Yc�A�%*

loss�<�<O��       �	����Yc�A�%*

loss ��<7�       �	�T��Yc�A�%*

lossi"o=��rg       �	���Yc�A�%*

loss��<��d       �	뫡�Yc�A�%*

lossO=�Y;       �	�O��Yc�A�%*

lossW=ѣx       �	N��Yc�A�%*

loss� �;���       �	����Yc�A�%*

losst =G/}`       �	����Yc�A�%*

loss��(<݇�       �	T��Yc�A�%*

loss��h<�Ǆ       �	�å�Yc�A�%*

loss���<A{�       �	sd��Yc�A�%*

loss3P=�O�       �	���Yc�A�%*

loss�+<[6�"       �	����Yc�A�%*

loss��!=8��J       �	�W��Yc�A�%*

loss���<��       �	�2��Yc�A�%*

lossÔv=�\	�       �	_Щ�Yc�A�%*

loss��<�D       �	�m��Yc�A�%*

loss=����       �	���Yc�A�%*

lossJt�<�Z�       �	r���Yc�A�%*

lossr�<��        �	Eh��Yc�A�%*

lossT�?<�ɘ�       �	%��Yc�A�%*

loss�o�=����       �	P���Yc�A�%*

loss�Tj<k�]�       �	!>��Yc�A�%*

loss�;<���i       �	�Ԯ�Yc�A�%*

loss��j=3v��       �	�n��Yc�A�%*

loss^��;�K%�       �	l��Yc�A�%*

lossϗ�<��]�       �	����Yc�A�%*

lossR6H<aq+3       �	�@��Yc�A�%*

losssZ=�`�%       �	����Yc�A�%*

lossty�=���       �	����Yc�A�%*

loss�%=ɉ�a       �	���Yc�A�%*

lossԌ=��(�       �	K���Yc�A�%*

loss��<s;7       �	�F��Yc�A�%*

lossf��<�G+�       �	����Yc�A�%*

loss��8<�Q��       �	�{��Yc�A�%*

loss(��=i���       �	f��Yc�A�%*

loss�U[=?��       �	���Yc�A�%*

loss_�E<��1M       �	�d��Yc�A�%*

loss���<��|Y       �	����Yc�A�%*

lossL],=	l��       �	%���Yc�A�%*

lossiF.=z�2       �	E*��Yc�A�%*

loss��
=���       �	����Yc�A�%*

loss�j�=����       �	Ϥ��Yc�A�%*

loss��;uks�       �	C��Yc�A�%*

loss ��<���       �	��Yc�A�%*

loss���<�#�       �	C���Yc�A�%*

loss�'j=�X\�       �	�C��Yc�A�%*

loss*<:�*k       �	����Yc�A�%*

loss��J=�[��       �	1z��Yc�A�%*

loss��<���]       �	o��Yc�A�%*

loss�Q=M3�       �	���Yc�A�%*

loss��k<T�.       �	�P��Yc�A�%*

loss�<�"�7       �	J���Yc�A�%*

loss�=�ԓ       �	u���Yc�A�%*

lossi3=d���       �	8,��Yc�A�%*

loss���<���       �	����Yc�A�%*

loss���<���1       �	���Yc�A�%*

lossT�<t��       �	�#��Yc�A�%*

loss�=^�[       �	 ���Yc�A�%*

loss��$<S�Ɉ       �	�q��Yc�A�%*

loss`">&���       �	�	��Yc�A�%*

loss��=2(�       �	o���Yc�A�%*

loss�4G=ؙy       �	�?��Yc�A�%*

loss�n<��       �	v���Yc�A�%*

lossљ�<�       �	n���Yc�A�%*

loss.�.=J�c�       �	#��Yc�A�%*

lossmP=��e       �	+���Yc�A�%*

loss�  =7�p       �	���Yc�A�%*

lossvCI<7�֢       �	|��Yc�A�%*

loss]`<{�t4       �	֧��Yc�A�%*

lossM��<9��{       �	S��Yc�A�%*

losse5= h       �	\��Yc�A�%*

loss)�$=6��       �	ؼ��Yc�A�%*

loss)�:=���-       �	Ze��Yc�A�%*

loss�>�<�1"E       �	��Yc�A�%*

loss��-=�P�       �	
��Yc�A�%*

loss�r ;J��       �	+n��Yc�A�%*

loss(f�<��U�       �	#��Yc�A�%*

loss}�/=����       �	y���Yc�A�%*

loss��<��Y�       �	WB��Yc�A�%*

loss�>?�O�       �	����Yc�A�%*

lossa*.=�^B&       �	\u��Yc�A�%*

lossi�;dV(       �	��Yc�A�%*

loss��>;�h)�       �	&���Yc�A�%*

loss[�x;)r��       �	�M��Yc�A�%*

loss�0G<4�ŀ       �	%t��Yc�A�%*

lossΗ=q2�4       �	���Yc�A�%*

lossJ��=���s       �	����Yc�A�%*

loss}ܖ<��t       �	�X��Yc�A�%*

lossj��;I�wc       �	���Yc�A�%*

loss���=��8�       �	���Yc�A�&*

loss��<�@?       �	(E��Yc�A�&*

loss��<�s�:       �	:���Yc�A�&*

loss}��<�3d       �	!���Yc�A�&*

loss���<Z?&�       �	R,��Yc�A�&*

loss�ʀ=+��       �	`���Yc�A�&*

lossS�o=�5k�       �	Me��Yc�A�&*

loss&T(=��m       �	����Yc�A�&*

loss��R=�c:-       �	����Yc�A�&*

loss
�+=�g��       �	�u��Yc�A�&*

lossR+7= <\�       �	���Yc�A�&*

lossx�g<����       �	����Yc�A�&*

lossۻ�=֡��       �	N��Yc�A�&*

loss =�-��       �	���Yc�A�&*

loss@]�<0E'�       �	���Yc�A�&*

lossG�<���       �	/��Yc�A�&*

loss_� =؅1       �	����Yc�A�&*

loss;�=-�S       �	2Y��Yc�A�&*

loss��Y=��>       �	_���Yc�A�&*

loss=9=�lw       �	� �Yc�A�&*

loss&��;q��v       �	�'�Yc�A�&*

loss8��<w�c�       �	���Yc�A�&*

loss���<���x       �	?��Yc�A�&*

loss?'+=-�+N       �	3�Yc�A�&*

loss��b=�do�       �	���Yc�A�&*

loss&z�<�S       �	kd�Yc�A�&*

lossڣ�<"�       �	j��Yc�A�&*

loss��=���        �	���Yc�A�&*

loss�!6=�A�       �	;m�Yc�A�&*

loss��<e^�       �	�X�Yc�A�&*

lossQ0�<Z#�       �	}^�Yc�A�&*

loss�.�=rS��       �	e	�Yc�A�&*

loss\$�;�:       �	��	�Yc�A�&*

loss�C0=�3       �	p�
�Yc�A�&*

loss죣<��#�       �	*t�Yc�A�&*

loss��=�);o       �	�.�Yc�A�&*

lossX5�<N�`�       �	g��Yc�A�&*

lossR�<��%�       �	 ��Yc�A�&*

loss���<� �       �	���Yc�A�&*

loss,S	=�7@       �	�w�Yc�A�&*

loss��<v/�       �	#�Yc�A�&*

loss��=]l,B       �	���Yc�A�&*

loss��=2c�       �	���Yc�A�&*

loss� %<h���       �	u:�Yc�A�&*

lossbm=0~��       �	7��Yc�A�&*

loss��='B��       �	Y��Yc�A�&*

loss�=��8�       �	�)�Yc�A�&*

lossป=Q��       �	���Yc�A�&*

lossfK�<����       �	�z�Yc�A�&*

lossQ�=���       �	�"�Yc�A�&*

loss��E<p|�^       �	���Yc�A�&*

loss���=��g�       �	ur�Yc�A�&*

loss�Z^<��v�       �	Y�Yc�A�&*

lossL=v�-       �	3��Yc�A�&*

lossҊH=��ޟ       �	�t�Yc�A�&*

loss���<���_       �	e�Yc�A�&*

lossj��;b�'_       �	��Yc�A�&*

loss1I=E�[       �	�q�Yc�A�&*

lossQ�=�x}       �	z�Yc�A�&*

loss�8]=7�z       �	��Yc�A�&*

loss�C�<|��i       �	�T�Yc�A�&*

lossZ��;���7       �	8��Yc�A�&*

loss���=�ŀK       �	�A�Yc�A�&*

losso�,<Z�{       �	���Yc�A�&*

lossf�Q<I�$@       �	�� �Yc�A�&*

losslFY=�41f       �	m6!�Yc�A�&*

loss��;=�>p�       �	��!�Yc�A�&*

loss��!=D��       �	��"�Yc�A�&*

lossF$"=��yS       �	�#�Yc�A�&*

loss/8<"��
       �	��#�Yc�A�&*

loss���<�P�       �	G$�Yc�A�&*

loss_�=��j       �	��$�Yc�A�&*

loss[0=���       �	�r%�Yc�A�&*

loss�J=<�Q]X       �	B&�Yc�A�&*

loss�g=6��F       �	8�&�Yc�A�&*

loss��<�&��       �	i6'�Yc�A�&*

loss$=ɬG�       �	_�'�Yc�A�&*

loss�l�;D!�       �	n(�Yc�A�&*

loss/��<�#��       �	�)�Yc�A�&*

loss�an=\�6�       �	P�)�Yc�A�&*

loss�mF=g���       �	�D*�Yc�A�&*

loss�R=%��c       �	��*�Yc�A�&*

losss��<��W�       �	�v+�Yc�A�&*

lossa�<���       �	�
,�Yc�A�&*

loss*<�;�	�Q       �	��,�Yc�A�&*

lossi�I=��       �	��-�Yc�A�&*

loss��<�f?       �	,.�Yc�A�&*

loss�\�<a-��       �	�5/�Yc�A�&*

loss֍=��       �	��/�Yc�A�&*

loss
�+=���
       �	Mj0�Yc�A�&*

loss�;�<��wH       �	�1�Yc�A�&*

lossLi�=�)�       �	��1�Yc�A�&*

loss��=<��E       �	/2�Yc�A�&*

loss�=Ǌ��       �	C�2�Yc�A�&*

loss�+=�D_*       �	$`3�Yc�A�&*

lossh��=�c�       �	3�3�Yc�A�&*

loss�͹<��=&       �	��4�Yc�A�&*

loss}�=:0�       �	d5�Yc�A�&*

loss#��<[���       �	�5�Yc�A�&*

loss��)=t��       �	q�6�Yc�A�&*

loss%�[;�|�       �	2;7�Yc�A�&*

loss=�*=Ie�       �	��7�Yc�A�&*

loss�!�<&�f�       �	�k8�Yc�A�&*

lossT�.<����       �	�9�Yc�A�&*

loss��<0Ѐ       �	?�9�Yc�A�&*

lossU{�<6f�2       �	J]:�Yc�A�&*

loss���<���       �	��:�Yc�A�&*

loss�@a=�~~�       �	u�;�Yc�A�&*

loss�l<���e       �	*<�Yc�A�&*

lossˎ<��5       �	r�<�Yc�A�&*

loss8�'=� j       �	�`=�Yc�A�&*

lossn�<�4       �	��=�Yc�A�&*

loss4��<�{�       �	ݶ>�Yc�A�&*

lossw�@=�	K�       �	PS?�Yc�A�&*

loss��L=�0w0       �	��?�Yc�A�&*

loss��<���5       �	$�@�Yc�A�&*

loss
�/=���-       �	iWA�Yc�A�&*

lossRiY<��       �	hyB�Yc�A�&*

lossH�;= �N       �	JC�Yc�A�&*

lossGH<��6<       �	��C�Yc�A�&*

loss�H�<p&g       �	�:D�Yc�A�&*

lossd1�<����       �	��D�Yc�A�&*

losso��<?d��       �	�hE�Yc�A�&*

loss�ML=b��       �	��G�Yc�A�&*

lossH�=��:�       �	�I�Yc�A�&*

lossmL-=���r       �	G�I�Yc�A�&*

loss(aP=�K@�       �	lK�Yc�A�&*

lossqk4=B>a       �	S?L�Yc�A�&*

loss��<��RZ       �	C�L�Yc�A�'*

loss%��<n�ñ       �	3N�Yc�A�'*

loss@�1<�7�(       �	b�N�Yc�A�'*

loss��<
���       �	��O�Yc�A�'*

lossJ�<Or��       �	�6P�Yc�A�'*

loss �<P'ڒ       �	�Q�Yc�A�'*

lossW�l=f�Gv       �	�Q�Yc�A�'*

loss�Qi=�ڶ       �	��R�Yc�A�'*

lossww�=��դ       �	�US�Yc�A�'*

lossW/=�QV       �	E�S�Yc�A�'*

loss~�<���       �	�T�Yc�A�'*

loss�d�<�g`       �	�rU�Yc�A�'*

losso�="���       �	RV�Yc�A�'*

lossX8�<��wv       �	ƧV�Yc�A�'*

loss��=O��       �	W�Yc�A�'*

loss��|=�k       �	#X�Yc�A�'*

loss:�=�tw�       �	*�X�Yc�A�'*

loss2?M=��n�       �	IY�Yc�A�'*

loss� v=T�K�       �	��Y�Yc�A�'*

loss�ع<�"<�       �	�yZ�Yc�A�'*

loss8��<��       �	.[�Yc�A�'*

losseI<w�:Q       �	��[�Yc�A�'*

lossޤ=��1       �	�N\�Yc�A�'*

loss}��=ޮ�:       �	��\�Yc�A�'*

loss?��=�*�       �	_}]�Yc�A�'*

lossZB=b��       �	�o^�Yc�A�'*

losss<�=J|�#       �	1_�Yc�A�'*

loss��6=o�f}       �	(�_�Yc�A�'*

loss��A=�'<       �	�5`�Yc�A�'*

loss��)<�uXl       �	��`�Yc�A�'*

losspw!<͘��       �	�aa�Yc�A�'*

loss}��<��<S       �	�b�Yc�A�'*

loss��q=�K�       �	#�b�Yc�A�'*

loss?�=Q�9       �	yd�Yc�A�'*

loss%�=�h       �	{�d�Yc�A�'*

loss��<j�}       �	��e�Yc�A�'*

loss)��<����       �	�{f�Yc�A�'*

loss�=@�g       �	�h�Yc�A�'*

lossf<x��       �	W�h�Yc�A�'*

loss3-L<�i��       �	CSi�Yc�A�'*

loss?<�}]3       �	F�i�Yc�A�'*

lossae�<B�F|       �	�j�Yc�A�'*

lossȔ�<�eF       �	�Yk�Yc�A�'*

loss��=^#       �	��k�Yc�A�'*

loss�"=g���       �	�m�Yc�A�'*

lossU�=�|$h       �	��m�Yc�A�'*

loss� =�3�       �	shn�Yc�A�'*

loss�=���
       �	o�Yc�A�'*

loss���<����       �	}�o�Yc�A�'*

loss��<�2�_       �	�Hp�Yc�A�'*

loss��=%�0�       �	�p�Yc�A�'*

loss�Yi=R�?F       �	��q�Yc�A�'*

loss$U�<b1��       �	�Dr�Yc�A�'*

loss=�f�       �	��r�Yc�A�'*

loss���<y�C       �	b�s�Yc�A�'*

loss�Ć<ߨt       �	t�Yc�A�'*

loss͎=/_Ɗ       �	(�t�Yc�A�'*

lossC �=���       �	XWu�Yc�A�'*

loss�y�<��I�       �	��u�Yc�A�'*

loss�1=^�f       �	ɮv�Yc�A�'*

loss���=��`�       �	��w�Yc�A�'*

lossqݷ<��68       �	Bx�Yc�A�'*

loss� n<�$.       �	��x�Yc�A�'*

loss��;שj       �	��y�Yc�A�'*

lossW�<��       �	pAz�Yc�A�'*

loss�6�<��       �	�z�Yc�A�'*

lossΘ�<�ҧ�       �	�v{�Yc�A�'*

lossj�=l�       �	(|�Yc�A�'*

loss7��<���       �	ک|�Yc�A�'*

losss\�<-�N�       �	�@}�Yc�A�'*

lossHJ=P��%       �	"�}�Yc�A�'*

lossf�i=#��       �	Q�~�Yc�A�'*

lossq0=��L       �	-!�Yc�A�'*

losswڤ=���&       �	2��Yc�A�'*

loss�hx<�i	�       �	D��Yc�A�'*

loss�H�=���       �	a���Yc�A�'*

loss��&=&�<       �	�z��Yc�A�'*

loss-�T=oe��       �	&��Yc�A�'*

loss��=F���       �	Mփ�Yc�A�'*

loss;�+=�9Y       �	�x��Yc�A�'*

loss3\<�3�       �	���Yc�A�'*

loss��q=�PaC       �	����Yc�A�'*

lossfp+=��Y'       �	)^��Yc�A�'*

lossq~�=�       �	���Yc�A�'*

loss���=���       �	b���Yc�A�'*

lossN:=hp�p       �	֌��Yc�A�'*

loss\�;;x�r{       �	kJ��Yc�A�'*

loss �M=�!�       �	���Yc�A�'*

loss��=�=+0       �	�X��Yc�A�'*

loss8��<y��       �	7Q��Yc�A�'*

loss)�<�t�       �	J���Yc�A�'*

loss^�!=�Œ�       �	���Yc�A�'*

lossϳ^<5�       �	8��Yc�A�'*

loss��}=���       �	�Ў�Yc�A�'*

loss��$=m�       �	�f��Yc�A�'*

lossfmy=�vb       �	"���Yc�A�'*

loss��<�mZ�       �	Ę��Yc�A�'*

loss��<�5�W       �	�.��Yc�A�'*

loss� =��6�       �	�ȑ�Yc�A�'*

lossZ�Q;�.�       �	Ja��Yc�A�'*

loss�]o=�=�5       �	v���Yc�A�'*

loss��;����       �	R���Yc�A�'*

losso�<NQ'o       �	`?��Yc�A�'*

loss8w<�o��       �	��Yc�A�'*

loss�[=��P�       �	���Yc�A�'*

loss��Q<fP�       �	���Yc�A�'*

loss��J=n��       �	����Yc�A�'*

loss��~=�t}       �	DP��Yc�A�'*

loss.Q�=����       �	���Yc�A�'*

loss2=QT�       �	t���Yc�A�'*

lossF��<�j��       �	e6��Yc�A�'*

loss�=��       �	���Yc�A�'*

loss,�.=� ��       �	iɚ�Yc�A�'*

lossr�=�xEK       �	�n��Yc�A�'*

loss�Aq=w�D�       �	��Yc�A�'*

loss&��;R%�       �	����Yc�A�'*

loss9�=��?�       �	�Q��Yc�A�'*

loss-�=L�2       �	���Yc�A�'*

loss=Y0=�`�       �	M���Yc�A�'*

lossȊ�<��)       �	��Yc�A�'*

lossCi�<�O       �	����Yc�A�'*

lossÙ=<JJu       �	�~��Yc�A�'*

lossl��<\��&       �	���Yc�A�'*

lossd)<���       �	����Yc�A�'*

loss-<�<�]͒       �	�O��Yc�A�'*

loss��<�R�       �	.���Yc�A�'*

lossQ�=7�=       �	�&��Yc�A�'*

loss�i-=jTW       �	yɤ�Yc�A�'*

loss}�<���|       �	8j��Yc�A�(*

loss3{�<����       �	6��Yc�A�(*

lossQ�	=� ��       �	���Yc�A�(*

lossR)?=�n�       �	����Yc�A�(*

loss�q==�]��       �	s+��Yc�A�(*

loss*'=mV�       �	%ɨ�Yc�A�(*

lossQ�<�       �	oe��Yc�A�(*

loss�R�<�G�       �	  ��Yc�A�(*

loss�$=]5�L       �	6ͪ�Yc�A�(*

loss�R=���       �	�c��Yc�A�(*

lossZ'H=:m��       �	� ��Yc�A�(*

loss��v<�L�g       �	���Yc�A�(*

loss��<�#T/       �	�6��Yc�A�(*

loss׃�;~&�J       �	Wͭ�Yc�A�(*

lossj6�<+xN�       �	�i��Yc�A�(*

loss<"�=�MK`       �	���Yc�A�(*

lossX�O=�Z�Y       �	훯�Yc�A�(*

lossֻ�<�
�       �	�3��Yc�A�(*

loss��<��R�       �	�ʰ�Yc�A�(*

loss�s=��߽       �	�i��Yc�A�(*

loss�%<��s       �	����Yc�A�(*

loss1��<�".W       �	v���Yc�A�(*

loss�"<��       �	�A��Yc�A�(*

loss��=g�!       �	�׳�Yc�A�(*

loss�e�<��g�       �	�r��Yc�A�(*

loss�C�=D>a�       �	�2��Yc�A�(*

loss��=#�"       �	ص�Yc�A�(*

loss%%�<���M       �	dv��Yc�A�(*

lossv(�<i���       �	3��Yc�A�(*

lossέ
=� *R       �	ݲ��Yc�A�(*

loss=��<N�f       �	��Yc�A�(*

loss���<��PL       �	�;��Yc�A�(*

loss*��<WL/�       �	�ݹ�Yc�A�(*

loss�@w<���       �	�{��Yc�A�(*

loss�=�r8       �	�'��Yc�A�(*

loss�#=Z�B       �	�Ż�Yc�A�(*

loss�w�;bK m       �	.���Yc�A�(*

losst�<�sW�       �	R,��Yc�A�(*

loss���<��Z�       �	�ؽ�Yc�A�(*

loss?��<,;       �	�q��Yc�A�(*

lossw��<�^&�       �	���Yc�A�(*

loss�h�<I٣�       �	���Yc�A�(*

loss��X=��       �	����Yc�A�(*

loss���=��?       �	E���Yc�A�(*

loss��=�l�       �	�&��Yc�A�(*

loss��<�z�       �	_���Yc�A�(*

loss8.r=5 �       �	�k��Yc�A�(*

lossFpi=��p       �	���Yc�A�(*

lossS��<r�٦       �	Ǜ��Yc�A�(*

loss�D<K�`       �	�2��Yc�A�(*

loss�L�<��\�       �	����Yc�A�(*

loss�)H=h�o�       �	t|��Yc�A�(*

loss`<=�T�d       �	e��Yc�A�(*

lossv�=��P       �	����Yc�A�(*

lossmv�<h�+       �	�V��Yc�A�(*

loss`Y]=��       �	���Yc�A�(*

loss�=r��8       �	=���Yc�A�(*

loss�B=p<�       �	����Yc�A�(*

loss��<�T�       �	�'��Yc�A�(*

lossz�0=�U�       �	���Yc�A�(*

lossj�=J>3       �	����Yc�A�(*

loss� =�'��       �	F��Yc�A�(*

lossWn=��h�       �	"���Yc�A�(*

lossנ;=a��U       �	�{��Yc�A�(*

loss���<�]��       �	v��Yc�A�(*

loss� �;��B�       �	"���Yc�A�(*

lossfU<�F�       �	����Yc�A�(*

loss�qX=�0�3       �	�D��Yc�A�(*

loss�l"=҃x       �	?��Yc�A�(*

loss�ZU<uq��       �	����Yc�A�(*

lossH�<ֵBz       �	Nb��Yc�A�(*

loss�=�4�       �	����Yc�A�(*

lossl��=��       �	R���Yc�A�(*

loss�r�=� /�       �	�4��Yc�A�(*

lossEق=��;s       �	U���Yc�A�(*

loss8\�<��       �	l���Yc�A�(*

lossx=�{       �	�o��Yc�A�(*

loss�8<U���       �	���Yc�A�(*

lossW�=Wf       �	n���Yc�A�(*

loss�<\P��       �	�D��Yc�A�(*

loss��=�
3�       �	����Yc�A�(*

loss�Z7<��h       �	`���Yc�A�(*

loss��_=�T��       �	a4��Yc�A�(*

losse/=*�8       �	���Yc�A�(*

loss�J�<����       �	v���Yc�A�(*

loss��w<L�h       �	D��Yc�A�(*

loss�o�<���       �	����Yc�A�(*

loss&��=Qf��       �	����Yc�A�(*

loss<v=7�l�       �	y#��Yc�A�(*

loss��<(�W       �	����Yc�A�(*

lossφ=oB��       �	���Yc�A�(*

loss2�=�L/�       �	[���Yc�A�(*

lossX�=^�n       �	$���Yc�A�(*

loss�n=�r��       �	���Yc�A�(*

lossʗ+=��       �	1��Yc�A�(*

loss��=���U       �	t���Yc�A�(*

loss�2U=���n       �	�v��Yc�A�(*

loss19<�6��       �	���Yc�A�(*

loss�R<J�/f       �	���Yc�A�(*

loss�<"=�v6�       �	�t��Yc�A�(*

loss�=�=[&��       �	f��Yc�A�(*

loss��=�� �       �	���Yc�A�(*

loss�D�<Q���       �	����Yc�A�(*

loss�ت<��T       �	'/��Yc�A�(*

lossq'c=`��c       �	f���Yc�A�(*

loss�u[=q3�b       �	і��Yc�A�(*

loss�g=��E       �	�2��Yc�A�(*

loss;��<#�$       �	����Yc�A�(*

loss�T=9B*       �	�m��Yc�A�(*

loss+�=Lh��       �		��Yc�A�(*

lossQ�<�K&3       �	V���Yc�A�(*

loss�W~=����       �	�V��Yc�A�(*

loss�d�<�!D       �	����Yc�A�(*

loss͓�<�SL@       �	����Yc�A�(*

loss�X�<ٛy�       �	(a��Yc�A�(*

loss,�2=F-`�       �	a���Yc�A�(*

loss�
�<�d%�       �	ė��Yc�A�(*

loss�T�<�gb�       �	8��Yc�A�(*

loss;�<h�	�       �	(���Yc�A�(*

loss���<
�       �	�|��Yc�A�(*

loss�"�<��o�       �	5%��Yc�A�(*

loss�܇=� u�       �	����Yc�A�(*

loss��}=�~�       �	�i��Yc�A�(*

loss�	�<�%�~       �	�
��Yc�A�(*

loss�p�<M��       �	����Yc�A�(*

loss�;L�V       �	�R��Yc�A�(*

loss�`O=��9       �	n���Yc�A�(*

loss˃>N��       �	ԙ��Yc�A�(*

lossh;�<��       �	?9��Yc�A�)*

loss�_=�|       �	3���Yc�A�)*

loss}��<Ӑ��       �	y���Yc�A�)*

loss�j=/h�N       �	�J��Yc�A�)*

loss���<K��       �	���Yc�A�)*

loss�LV<��y>       �	S���Yc�A�)*

loss�L=��zz       �	Id��Yc�A�)*

lossƁ=~�       �	E/��Yc�A�)*

loss&~=)?�       �	-���Yc�A�)*

loss An=��       �	&m �Yc�A�)*

loss���<(��       �	��Yc�A�)*

losse�<���       �	��Yc�A�)*

loss�]�=�� �       �	TR�Yc�A�)*

loss�&=בe�       �	���Yc�A�)*

lossnPT<]K��       �	���Yc�A�)*

lossq.�;       �	�(�Yc�A�)*

loss�T*=}9�@       �		��Yc�A�)*

lossZ��<=�ܠ       �	�_�Yc�A�)*

loss_�]=wo*?       �	���Yc�A�)*

loss��;�D_w       �	���Yc�A�)*

loss��<H��       �	�N�Yc�A�)*

loss�xM=�˾�       �	���Yc�A�)*

loss��<-q��       �	H��Yc�A�)*

lossH��=���       �	5A	�Yc�A�)*

loss�0=
O�P       �	��	�Yc�A�)*

lossFǲ=��{�       �	�
�Yc�A�)*

lossd
C=�5�       �	��Yc�A�)*

loss��<Պ>J       �	���Yc�A�)*

loss�;ʥ/�       �	O�Yc�A�)*

loss�(<.���       �	���Yc�A�)*

loss�i= '�       �	~��Yc�A�)*

loss��<��8p       �	�/�Yc�A�)*

loss-�A=����       �	F��Yc�A�)*

loss���=��,�       �	�1�Yc�A�)*

lossۊ =nr��       �	���Yc�A�)*

lossx�=���       �	o�Yc�A�)*

loss��=�{�9       �	��Yc�A�)*

lossɑ�<�	I@       �	���Yc�A�)*

loss�l�<�;�       �	b�Yc�A�)*

loss��<�?�       �	m�Yc�A�)*

loss��<Dw       �	���Yc�A�)*

loss�;�<3�       �	�B�Yc�A�)*

loss#=�B-�       �	���Yc�A�)*

loss���=�ۺ�       �	ty�Yc�A�)*

loss���<8���       �	7�Yc�A�)*

loss�n<�I�       �	g��Yc�A�)*

loss�?�<��ݐ       �	?X�Yc�A�)*

loss��<ZrM       �	S�Yc�A�)*

loss=@�<�ɒ       �	~��Yc�A�)*

loss�VH==�~       �	<K�Yc�A�)*

loss
`�<��h�       �	!��Yc�A�)*

lossg�<f�i�       �	���Yc�A�)*

loss�Z=����       �	r3�Yc�A�)*

loss�]=��/�       �	x��Yc�A�)*

loss�<OF7�       �	��Yc�A�)*

lossB <����       �	F$�Yc�A�)*

loss�4=��\       �	b��Yc�A�)*

loss��<OP��       �	�b�Yc�A�)*

lossp�=��-       �	���Yc�A�)*

lossW!�<�-`       �	z� �Yc�A�)*

loss��<=��O�       �	.S!�Yc�A�)*

loss�c=���       �	� "�Yc�A�)*

loss��==��R       �	;�"�Yc�A�)*

loss��<<�       �	mT#�Yc�A�)*

loss?��=��       �	 �#�Yc�A�)*

loss���;Ʋ�?       �	��$�Yc�A�)*

lossq��=A��       �	v5%�Yc�A�)*

loss<��=����       �	��%�Yc�A�)*

loss��<:Ԥ}       �	f�&�Yc�A�)*

loss���<�32�       �	�''�Yc�A�)*

lossW=2��       �	y�'�Yc�A�)*

lossvu<��       �	
h(�Yc�A�)*

loss�4=@ĩ�       �	�)�Yc�A�)*

lossV{D<��3N       �	��)�Yc�A�)*

loss��+<��u�       �	�D*�Yc�A�)*

loss&�L<���       �	��*�Yc�A�)*

lossȭ�<�A�       �	�}+�Yc�A�)*

loss�|=f6�       �	,�Yc�A�)*

loss�<�H�X       �	w�,�Yc�A�)*

loss�<>	��       �	l]-�Yc�A�)*

lossZ#=�
��       �	��-�Yc�A�)*

loss���=��dl       �	�.�Yc�A�)*

loss4�z<Dg��       �	�-/�Yc�A�)*

loss\]�;$��       �	��/�Yc�A�)*

loss�	:=��'       �	bh0�Yc�A�)*

loss\-�;�i       �	�1�Yc�A�)*

loss��*<�Mq�       �	1�1�Yc�A�)*

loss��;�+)~       �	?V2�Yc�A�)*

loss|z�;�c*       �	U�2�Yc�A�)*

lossdQ�<w�k
       �	��3�Yc�A�)*

loss���<�A�H       �	D�4�Yc�A�)*

loss�N)<s�S�       �	>%5�Yc�A�)*

loss�|=�K�f       �	��5�Yc�A�)*

loss[��;��R[       �	#h6�Yc�A�)*

loss�9T��       �	67�Yc�A�)*

lossf�;0�*K       �	ˢ7�Yc�A�)*

loss�<��k       �	�D8�Yc�A�)*

loss�� =�7��       �	��8�Yc�A�)*

loss<�2��       �	@�9�Yc�A�)*

loss\�a;׆ko       �	��:�Yc�A�)*

lossr�<��z�       �	v�;�Yc�A�)*

loss��>�S9�       �	�*<�Yc�A�)*

loss�փ;s䥬       �	,�<�Yc�A�)*

lossf�i<'��7       �	;r=�Yc�A�)*

lossw�'=��A       �	�>�Yc�A�)*

loss�Ƌ=�Ƚ�       �	l�>�Yc�A�)*

lossz�h<�j�b       �	��?�Yc�A�)*

loss ))<�-��       �	/5@�Yc�A�)*

loss�.=#��}       �	g�@�Yc�A�)*

lossD�=N��       �	*qA�Yc�A�)*

loss�?x=3Ѵ�       �	AB�Yc�A�)*

lossh��=2�T?       �	��B�Yc�A�)*

loss���<J�q�       �	�PC�Yc�A�)*

loss�=h=f���       �	$�C�Yc�A�)*

loss��="���       �	 �D�Yc�A�)*

loss�Q	=���       �	G=E�Yc�A�)*

lossH�+=��Խ       �	<�E�Yc�A�)*

lossM��=S��]       �	|�F�Yc�A�)*

loss��D=h!�       �	�%G�Yc�A�)*

loss�,�<���       �	��G�Yc�A�)*

loss�=���}       �	�fH�Yc�A�)*

lossϛ=�~/       �	gcI�Yc�A�)*

loss��;=O�w       �	��J�Yc�A�)*

loss�G=�1ߍ       �	SL�Yc�A�)*

loss�;<37�       �	ʧL�Yc�A�)*

loss�CP;Th��       �	�HM�Yc�A�)*

lossZ��<��jq       �	��M�Yc�A�)*

loss��;���[       �	��N�Yc�A�)*

loss1_�<�H�       �	�OO�Yc�A�**

loss�r<��       �	"�O�Yc�A�**

loss�s?==��)       �	��P�Yc�A�**

loss=�9=�<Nt       �	��Q�Yc�A�**

loss��;լ��       �	�:R�Yc�A�**

lossc <Hph       �	��R�Yc�A�**

loss�Hy<|�Y4       �	x}S�Yc�A�**

loss��E<���       �	T�Yc�A�**

loss�] =��(�       �	��T�Yc�A�**

loss(0�<���       �	^U�Yc�A�**

loss=�m<d�       �	i V�Yc�A�**

loss�Ɂ=�D       �	^�V�Yc�A�**

loss�m�=�j       �	WZW�Yc�A�**

loss�D=��       �	�W�Yc�A�**

loss�Q�;EO�       �	��X�Yc�A�**

loss�zX<�0Q       �	�OY�Yc�A�**

losso�"<*s�{       �	�Y�Yc�A�**

lossġ�<�RK�       �	��Z�Yc�A�**

lossI�;�       �	�2[�Yc�A�**

lossn@=��2x       �	��[�Yc�A�**

loss���<�8��       �	Y�\�Yc�A�**

loss�F�;![w       �	�"]�Yc�A�**

loss�
W=�0�S       �	��]�Yc�A�**

loss��<��       �	3p^�Yc�A�**

lossRl�<2BFo       �	�_�Yc�A�**

loss�V�<-�g       �	%�x�Yc�A�**

loss�&8=PJ�8       �	��y�Yc�A�**

lossd�k=2o�}       �	�*z�Yc�A�**

losso�<�;�:       �	"�z�Yc�A�**

loss���<�        �	rk{�Yc�A�**

lossJ��<�1�       �	�|�Yc�A�**

loss���<A3       �	V�|�Yc�A�**

lossJ*^<��       �	)^}�Yc�A�**

loss��=��6       �	�~�Yc�A�**

loss�A=d�(       �	��~�Yc�A�**

loss�+�;�W�       �	à�Yc�A�**

loss�K_=���       �	�F��Yc�A�**

loss =��r       �	>��Yc�A�**

loss�%�=�I!@       �	����Yc�A�**

loss��W=�*Q�       �	9'��Yc�A�**

loss):�=l�H       �	�΂�Yc�A�**

lossĶ;�C��       �	�r��Yc�A�**

loss���<���       �	b��Yc�A�**

loss��< �<       �	��Yc�A�**

loss���<T!��       �	�g��Yc�A�**

loss
�`<��eI       �	c
��Yc�A�**

loss{��<���       �	7���Yc�A�**

loss��2<�!��       �	�J��Yc�A�**

loss4�W=s�X       �	��Yc�A�**

loss���<�׌       �	�{��Yc�A�**

loss7�E<;��       �	~���Yc�A�**

loss~P�<c-�       �	��Yc�A�**

loss�=w)�R       �	����Yc�A�**

lossj��<�ѕ=       �	���Yc�A�**

loss��+=C�'       �	����Yc�A�**

loss�<=欂�       �	�f��Yc�A�**

lossj�<ޒE       �	���Yc�A�**

loss�5 <� ��       �	���Yc�A�**

loss@�#=Wm       �	�J��Yc�A�**

lossqT�<�Z&       �	���Yc�A�**

lossJ=(П       �	 ���Yc�A�**

lossؾ;��C�       �	#2��Yc�A�**

loss�a�=v�d       �	�Α�Yc�A�**

loss8d�=�i�       �	�h��Yc�A�**

loss�� =�'�       �	F��Yc�A�**

loss\N�<�L�       �	�̓�Yc�A�**

loss|2<�	��       �	�e��Yc�A�**

loss�y<U{�       �	:��Yc�A�**

lossH�M=Z�       �	MK��Yc�A�**

loss=g�<=Z-,       �	P��Yc�A�**

loss���<�fxw       �	R~��Yc�A�**

loss6Dt=U.|�       �	% ��Yc�A�**

loss�ӓ=�S       �	Ը��Yc�A�**

loss�Xh=(?;�       �	�P��Yc�A�**

loss��a;�[I8       �	���Yc�A�**

loss6#<Ī��       �	����Yc�A�**

loss��l=2�-�       �	���Yc�A�**

loss�`=A�I�       �	����Yc�A�**

lossNN>��       �	�T��Yc�A�**

loss=ʆ<R�S�       �	����Yc�A�**

loss���<C�]�       �	����Yc�A�**

loss�4;��       �	0��Yc�A�**

loss_�!;�v8�       �	O̞�Yc�A�**

loss<�z<�r       �	����Yc�A�**

lossxE=�L       �	F'��Yc�A�**

loss��=X�g(       �	�Š�Yc�A�**

loss�I�<�>�       �	X���Yc�A�**

lossĖg;K�P�       �	W%��Yc�A�**

loss�o=��2�       �	+���Yc�A�**

lossa�<f,       �	T��Yc�A�**

loss�2<�       �	����Yc�A�**

lossS�Q=���Z       �	Ϟ��Yc�A�**

loss�{�<t0#       �	�x��Yc�A�**

loss�[==.4�       �	j.��Yc�A�**

loss��c=�)��       �	�Ϧ�Yc�A�**

loss�t%=�1_�       �	�o��Yc�A�**

loss_��<ڬ�       �	���Yc�A�**

lossR0=��Yg       �	L���Yc�A�**

loss�hT=5�       �	�N��Yc�A�**

loss��<�
��       �	���Yc�A�**

loss��0<a�b�       �	Ӈ��Yc�A�**

loss���;_�M       �	�$��Yc�A�**

losslp�<��̙       �	`ɫ�Yc�A�**

loss@Qf<l2�k       �	�n��Yc�A�**

loss�]<,hqP       �	��Yc�A�**

lossc =Cfo       �	����Yc�A�**

loss��=�L.�       �	N`��Yc�A�**

loss�F�<�G�       �	-��Yc�A�**

losso�;�>g�       �	$���Yc�A�**

loss�q�<K'%       �	�d��Yc�A�**

loss��<~�       �	)��Yc�A�**

loss��<K"Oj       �	����Yc�A�**

lossN��<�_Y�       �	�Y��Yc�A�**

lossɾ�<���       �	����Yc�A�**

loss��<ې��       �	A���Yc�A�**

loss�.=���       �	1?��Yc�A�**

loss:|=^ �       �	d��Yc�A�**

loss��)=>)�       �	%���Yc�A�**

loss�m(=�l�       �	?��Yc�A�**

loss�%=s�       �	���Yc�A�**

loss�w<'�[6       �	����Yc�A�**

loss�+�<@�3�       �	&��Yc�A�**

loss���<��܀       �	� ��Yc�A�**

loss(=o��       �	|���Yc�A�**

loss�\=>�PN       �	+l��Yc�A�**

loss&<���       �	O��Yc�A�**

loss;~S<�%N�       �	6Ȼ�Yc�A�**

loss�=���       �	�l��Yc�A�**

loss@ֺ<�*�V       �	Y��Yc�A�+*

loss�=�$��       �	Pǽ�Yc�A�+*

loss��<1��2       �	�s��Yc�A�+*

losslڶ<���       �	d>��Yc�A�+*

loss&�=�h�h       �	Uܿ�Yc�A�+*

loss���=�n�       �	�z��Yc�A�+*

loss,�O<����       �	�!��Yc�A�+*

loss���=cm|�       �	����Yc�A�+*

loss1�$=}���       �	v��Yc�A�+*

loss�=ަ�       �	��Yc�A�+*

loss���;B:��       �	���Yc�A�+*

loss��=Ic5       �	�X��Yc�A�+*

loss��@;d��       �	p��Yc�A�+*

loss�t�<��U       �	h���Yc�A�+*

lossԴ�<���F       �	�c��Yc�A�+*

loss���<T�x/       �	���Yc�A�+*

loss�f�<(��:       �	����Yc�A�+*

loss��<,�t       �	1]��Yc�A�+*

lossSq]=���v       �	2 ��Yc�A�+*

loss��=~�7�       �	����Yc�A�+*

lossC��<�}��       �	���Yc�A�+*

loss(N�;8��       �	X��Yc�A�+*

loss,n�=�B{�       �	���Yc�A�+*

lossߤ=�;B;       �	w���Yc�A�+*

loss��=&��6       �	���Yc�A�+*

loss�=O_7�       �	�3��Yc�A�+*

loss�� =�1��       �	����Yc�A�+*

loss:4�<��       �	 W��Yc�A�+*

loss*¤=JJv       �	����Yc�A�+*

loss��<��\�       �	����Yc�A�+*

loss�<�2��       �	Z-��Yc�A�+*

loss��=hqJ       �	����Yc�A�+*

loss�~�<J=�       �	�h��Yc�A�+*

loss1��<+C�%       �	���Yc�A�+*

loss��#=��[[       �	����Yc�A�+*

loss� �=?Z%       �	�?��Yc�A�+*

loss��"=�e�       �	����Yc�A�+*

lossv�<;�gr       �	T���Yc�A�+*

loss?1�<��4       �	'L��Yc�A�+*

loss`��<�i       �	!���Yc�A�+*

loss$=��       �	=���Yc�A�+*

loss�ޣ=m��       �	���Yc�A�+*

loss��@=���       �	����Yc�A�+*

loss���<c{�N       �	�`��Yc�A�+*

lossm`.;�P��       �	����Yc�A�+*

loss��A<�#�       �	���Yc�A�+*

lossĮ<����       �	d=��Yc�A�+*

lossƟ>=��"�       �	� ��Yc�A�+*

loss���=�%       �	U���Yc�A�+*

loss�G�=A�P�       �	:[��Yc�A�+*

loss�<�Z(       �	f���Yc�A�+*

lossЭ�=^�5x       �	���Yc�A�+*

loss��<�^�}       �	9b��Yc�A�+*

loss��=]��[       �	����Yc�A�+*

lossX��<Ђ�N       �	����Yc�A�+*

lossC#=�d��       �	,g��Yc�A�+*

lossVd�<����       �	J��Yc�A�+*

loss�P�<+�^       �	&���Yc�A�+*

loss|�W<����       �	�I��Yc�A�+*

loss2�=�<��       �	i��Yc�A�+*

loss�2�;�R       �	
���Yc�A�+*

loss�~�<� }       �	|���Yc�A�+*

loss�\<���       �	�*��Yc�A�+*

loss]¤;Z2�       �	2���Yc�A�+*

loss8 �<K�U       �	Bx��Yc�A�+*

loss��T=b۴�       �	��Yc�A�+*

lossY\�;'}>�       �	w���Yc�A�+*

lossi	Y=K���       �	B[��Yc�A�+*

loss �-<�ܧN       �	�1��Yc�A�+*

loss�>�<��D       �	:���Yc�A�+*

loss1I=.��h       �	|b��Yc�A�+*

loss���<(�j�       �	3���Yc�A�+*

lossu�<N^��       �	����Yc�A�+*

lossѿ=o)       �	D��Yc�A�+*

loss�=�R�`       �	���Yc�A�+*

loss
�<) ��       �	���Yc�A�+*

loss�8�<��i       �	Bx��Yc�A�+*

loss�6;=C?�       �	���Yc�A�+*

loss���<Y�u&       �	����Yc�A�+*

loss�D�;�V�       �	�w��Yc�A�+*

loss!��<`��       �	i��Yc�A�+*

loss��&<$�`       �	Ӽ��Yc�A�+*

loss�j<]{s�       �	�`��Yc�A�+*

loss�&=;���       �	]��Yc�A�+*

loss:~=�(I       �	v���Yc�A�+*

loss�YK=<��`       �	)���Yc�A�+*

loss8L=0YB/       �	�1��Yc�A�+*

loss���<[v+       �	����Yc�A�+*

loss��<I,�I       �	{��Yc�A�+*

loss��$<͓@�       �	j��Yc�A�+*

losswȏ<3VZW       �	>���Yc�A�+*

lossn[S<y�        �	�^��Yc�A�+*

lossd<,�
t       �	^���Yc�A�+*

loss�k�<�'��       �	2���Yc�A�+*

loss��=� ~�       �	�H��Yc�A�+*

loss�P?=�.�       �	��Yc�A�+*

loss�7W<��u       �	S���Yc�A�+*

loss��=Q@��       �	M��Yc�A�+*

lossΗ=���       �	����Yc�A�+*

loss�u<��}\       �	����Yc�A�+*

loss|H�=�r$�       �	
+ �Yc�A�+*

loss�Q<��       �	�� �Yc�A�+*

lossm|u= ���       �	܂�Yc�A�+*

loss�k�<YΏ/       �	�#�Yc�A�+*

loss��=��       �	r��Yc�A�+*

loss�J=����       �	��Yc�A�+*

loss��<V��       �	z6�Yc�A�+*

loss�;�<���,       �	���Yc�A�+*

loss}AG<�<O       �	���Yc�A�+*

loss��@<p(�       �	�0�Yc�A�+*

loss�<���       �	B��Yc�A�+*

lossP�=��,<       �	B��Yc�A�+*

loss}@,==x=       �	�K�Yc�A�+*

lossn��<����       �	�	�Yc�A�+*

loss��F=�(�@       �	��	�Yc�A�+*

loss�O�=�4)p       �	
�Yc�A�+*

loss���=��t       �	�0�Yc�A�+*

lossr�&=bZ�       �	�#�Yc�A�+*

lossc�
<�R�       �	���Yc�A�+*

loss�x =�v       �	Dk�Yc�A�+*

loss;[�=S5��       �	#0�Yc�A�+*

lossT��<��˽       �	��Yc�A�+*

loss���<�^�       �	���Yc�A�+*

losse�<[۸J       �	��Yc�A�+*

losso��<R*       �	�}�Yc�A�+*

loss�K<���       �	g��Yc�A�+*

lossH<�q��       �	dZ�Yc�A�+*

loss�`<�n/)       �	�x�Yc�A�+*

lossl�X<#�O�       �	r�Yc�A�,*

loss�@0=�WD]       �	=��Yc�A�,*

loss�m�<8r^�       �	�W�Yc�A�,*

lossM�
=xIE'       �	���Yc�A�,*

loss�AV=/g�O       �	2��Yc�A�,*

loss���<J� A       �	�G�Yc�A�,*

lossx$=��!       �	��Yc�A�,*

loss��W=N	_       �	m�Yc�A�,*

lossx��;���       �	 ��Yc�A�,*

loss�wT<�0��       �	^�Yc�A�,*

lossf�=i�       �	[$�Yc�A�,*

loss��'=Q�1�       �	v��Yc�A�,*

loss�O?=g�\q       �	-_�Yc�A�,*

lossNd�<��
       �	K�Yc�A�,*

loss��<ϜMk       �	Χ�Yc�A�,*

loss��<�cK�       �	�H �Yc�A�,*

loss��f<g�*       �	\!�Yc�A�,*

loss �-=��'?       �	ݚ!�Yc�A�,*

loss�ө<��       �	�4"�Yc�A�,*

loss�B�<�oҎ       �	O�"�Yc�A�,*

lossf�z<�!�g       �	b�#�Yc�A�,*

loss��=�g �       �	�6$�Yc�A�,*

lossl :<U��       �	{�$�Yc�A�,*

loss�׬;�'�&       �	?s%�Yc�A�,*

loss#"=�p�       �	&�Yc�A�,*

loss�*�<s ��       �	��&�Yc�A�,*

loss1ǅ<�)�A       �	"U'�Yc�A�,*

loss�*<B�'       �	��'�Yc�A�,*

lossєN=:���       �	N�(�Yc�A�,*

lossv�"=B�^       �	K:)�Yc�A�,*

lossJ�Q<,c.�       �	0�)�Yc�A�,*

lossZ��<9([�       �	��*�Yc�A�,*

loss�9�=�8�4       �	�I+�Yc�A�,*

lossx�=C���       �	
�+�Yc�A�,*

loss_��<�eW       �	Ș,�Yc�A�,*

loss�l�=��F�       �	:-�Yc�A�,*

lossC�<]#7       �	.�Yc�A�,*

lossOn=���       �	��.�Yc�A�,*

lossH��<b���       �	�[/�Yc�A�,*

lossf*=�1/�       �	�0�Yc�A�,*

loss���<$&��       �	�0�Yc�A�,*

loss�4=��d{       �	\1�Yc�A�,*

loss��=�]       �	�2�Yc�A�,*

loss
f�=�ٶ�       �	>�2�Yc�A�,*

loss�=D�ć       �	�T3�Yc�A�,*

lossra=��Cn       �	�3�Yc�A�,*

loss<�;K&E�       �	j�4�Yc�A�,*

loss�Cz=܆�       �	vp5�Yc�A�,*

lossrv=-5�!       �	N6�Yc�A�,*

loss�N�<ɃM       �	�6�Yc�A�,*

losst�<��w       �	nO7�Yc�A�,*

lossI�Q==��k       �	J�7�Yc�A�,*

loss�I	=��ߠ       �	�9�Yc�A�,*

loss%=�%�       �	�9�Yc�A�,*

lossnZ=�w��       �	�\:�Yc�A�,*

loss�F�<l���       �	�:�Yc�A�,*

loss	��<�a(0       �	�;�Yc�A�,*

lossw�.=�       �	><�Yc�A�,*

lossGU;��N-       �	A�<�Yc�A�,*

lossQ�;�S       �	�p=�Yc�A�,*

loss�C�=��K�       �	�>�Yc�A�,*

loss&X�<g0��       �	Φ>�Yc�A�,*

loss6<�O�u       �	�B?�Yc�A�,*

loss�Co<e�       �	>�?�Yc�A�,*

loss�Q3=|~Y�       �	a�@�Yc�A�,*

loss�3�<�G�+       �	�6A�Yc�A�,*

loss`�Q=^��&       �	�A�Yc�A�,*

loss	��=��       �	2vB�Yc�A�,*

loss�K=��1�       �	�C�Yc�A�,*

loss=�<a�Ӿ       �	��C�Yc�A�,*

loss��=�՝I       �	*RD�Yc�A�,*

loss�s�<����       �	��D�Yc�A�,*

loss��;=n9}A       �	i�E�Yc�A�,*

loss�Ν<-1�R       �	9+F�Yc�A�,*

losstP=�l�R       �	��F�Yc�A�,*

loss�X�;i�w       �	�jG�Yc�A�,*

loss^W=�]y�       �	�H�Yc�A�,*

loss1�<�@�]       �	��H�Yc�A�,*

loss�X�<�B$       �	a5I�Yc�A�,*

loss�
=널�       �	x�I�Yc�A�,*

loss�Xv=G"o       �	�eJ�Yc�A�,*

loss��=��us       �	�K�Yc�A�,*

lossl֡<ѡ       �	��K�Yc�A�,*

loss���<�m��       �	[[L�Yc�A�,*

loss(�q=c�7�       �	�L�Yc�A�,*

loss�pO=�kq�       �	��M�Yc�A�,*

lossz^<��k�       �	pxN�Yc�A�,*

loss�==�UC�       �	O�Yc�A�,*

lossWx�<�k�       �	��O�Yc�A�,*

lossM��<��]�       �	[?P�Yc�A�,*

loss�P=G�F�       �	��P�Yc�A�,*

loss@�=���,       �	�iQ�Yc�A�,*

loss��<?>|�       �	CR�Yc�A�,*

loss�'�<=�       �	�T�Yc�A�,*

loss��O=>��       �	��T�Yc�A�,*

loss���<]��=       �	��U�Yc�A�,*

losst�=`~H       �	S!V�Yc�A�,*

lossG5=D>��       �	]�V�Yc�A�,*

lossZ)E=GO��       �	�aW�Yc�A�,*

loss
�?=_țU       �	MX�Yc�A�,*

loss}�j<k���       �	)�X�Yc�A�,*

loss��d<��x�       �	n�Y�Yc�A�,*

loss��<��`�       �	/Z�Yc�A�,*

loss�ȗ=�Ћ�       �	��Z�Yc�A�,*

loss���<�ZD�       �	Qi[�Yc�A�,*

loss��u=�e�c       �	�\�Yc�A�,*

lossZ�<ᣈ�       �	 �\�Yc�A�,*

loss�,&=��&�       �	�Q]�Yc�A�,*

loss/M^;u��h       �	
�]�Yc�A�,*

loss�;N��g       �	z�^�Yc�A�,*

loss׫=��	�       �	*_�Yc�A�,*

loss#�;<�4\I       �	��_�Yc�A�,*

loss�~�<�w��       �	Dn`�Yc�A�,*

lossۛ�=I�       �	ta�Yc�A�,*

loss)$O=$q��       �	H�a�Yc�A�,*

loss�z =�<��       �	��b�Yc�A�,*

loss��=�T�       �	c�c�Yc�A�,*

loss>v"=�N�{       �	$Fd�Yc�A�,*

loss{�<�{�~       �	��d�Yc�A�,*

lossi@�<L3�T       �	�e�Yc�A�,*

loss��:=�;��       �	D�f�Yc�A�,*

loss)��<�bm       �	$}g�Yc�A�,*

lossu2�<��"F       �	�h�Yc�A�,*

loss�,�<�|�       �	�Pi�Yc�A�,*

loss���;��m       �	��i�Yc�A�,*

lossx�s<2�Q       �	��j�Yc�A�,*

loss��S<�q@�       �	�Cm�Yc�A�,*

loss�ui<A4v       �	-�m�Yc�A�,*

loss�r�<u��z       �	R�n�Yc�A�-*

loss��='=�       �	:\o�Yc�A�-*

loss/��<�Xd�       �	�o�Yc�A�-*

loss�Y=�5�       �	�p�Yc�A�-*

loss��j=�<�       �	�>q�Yc�A�-*

lossc��<+��       �	��r�Yc�A�-*

loss�@=9�C�       �	p?s�Yc�A�-*

loss�=�Gu}       �	��s�Yc�A�-*

loss�b�<h�UJ       �	�ut�Yc�A�-*

lossэ�<��r       �	W$u�Yc�A�-*

lossF<���       �	:�u�Yc�A�-*

loss}�=vwM�       �	*rv�Yc�A�-*

loss�W�;���Z       �	Gw�Yc�A�-*

loss�� =��4�       �	y�w�Yc�A�-*

loss3�<���       �	�hx�Yc�A�-*

loss��=���       �	Ey�Yc�A�-*

loss�NV=�~�K       �	��y�Yc�A�-*

losslU#=���       �	eUz�Yc�A�-*

losse�=@z�       �	��z�Yc�A�-*

loss-EK=�o�       �	�{�Yc�A�-*

loss(��=��<�       �	�7|�Yc�A�-*

loss?3�<��1�       �	�|�Yc�A�-*

loss��<�]�       �	�k}�Yc�A�-*

loss\��;�|�       �	�~�Yc�A�-*

loss���<�g��       �	��~�Yc�A�-*

loss�2�;�z�       �	�>�Yc�A�-*

loss}��;i�d=       �	���Yc�A�-*

loss�=;�e�       �	iq��Yc�A�-*

loss�zL=�tI       �	���Yc�A�-*

loss�D�;#�C       �	���Yc�A�-*

loss(�5=��V�       �	R��Yc�A�-*

lossc��<2�?       �	���Yc�A�-*

loss�/�<62f       �	H���Yc�A�-*

loss��w<"˔        �	|(��Yc�A�-*

loss�|�<Z��7       �	�ӄ�Yc�A�-*

lossz�<���       �	nn��Yc�A�-*

lossL\$=���       �	���Yc�A�-*

loss7��;�b�       �	����Yc�A�-*

loss�[w<o��       �	iT��Yc�A�-*

loss���<Ϊk       �	����Yc�A�-*

lossE(�<��p�       �	t���Yc�A�-*

loss�<�"�       �	�0��Yc�A�-*

loss��<SwA       �	�S��Yc�A�-*

loss�=?հ�       �	���Yc�A�-*

loss��=.�?q       �	����Yc�A�-*

loss�}:<W_�       �	S%��Yc�A�-*

loss���<�nT�       �	w���Yc�A�-*

lossv-p=��4       �	 V��Yc�A�-*

loss��!=�ta�       �	����Yc�A�-*

loss�i=�:_�       �	C���Yc�A�-*

loss
�g=�9�c       �	0I��Yc�A�-*

loss)%,=o>�       �	���Yc�A�-*

loss��5=��J�       �	4���Yc�A�-*

loss�t=��       �	���Yc�A�-*

lossμL=�/��       �	���Yc�A�-*

loss��=p28       �	I���Yc�A�-*

loss��0=m��
       �	G��Yc�A�-*

lossL/<a�       �	[���Yc�A�-*

losse=w�(�       �	@P��Yc�A�-*

loss� =���X       �	%��Yc�A�-*

loss�K�=^\J       �	r���Yc�A�-*

lossޣ<��A�       �	]��Yc�A�-*

loss�y�<�|�       �	����Yc�A�-*

lossҷ<ov��       �	:���Yc�A�-*

lossXk�=��j�       �	�K��Yc�A�-*

loss�eF=�i��       �	���Yc�A�-*

loss3�G=7��[       �	�}��Yc�A�-*

loss9�=�V7T       �	���Yc�A�-*

loss}7�=跬a       �	M���Yc�A�-*

lossL�=��S       �	#i��Yc�A�-*

loss�r�<���       �	���Yc�A�-*

loss��==[��       �	����Yc�A�-*

loss��S=�bf�       �	jP��Yc�A�-*

lossގ<�5J�       �	����Yc�A�-*

lossxm�<�M=3       �	p���Yc�A�-*

loss���<P��       �	�5��Yc�A�-*

loss�֎=f��        �	̟�Yc�A�-*

loss��=�:�       �	5b��Yc�A�-*

loss�F%=n�	�       �	W>��Yc�A�-*

loss%��<E���       �	���Yc�A�-*

loss��<j��1       �	����Yc�A�-*

loss���=��v�       �		6��Yc�A�-*

lossR��=��N       �	�ԣ�Yc�A�-*

loss��<�5       �	0���Yc�A�-*

loss}'�<�p^       �	�h��Yc�A�-*

loss�2�;���       �	�(��Yc�A�-*

loss3�N=IQ�       �	٦�Yc�A�-*

lossh�=2m       �	k~��Yc�A�-*

lossEm�<��       �	����Yc�A�-*

loss�|�<�5kH       �	%[��Yc�A�-*

loss���<*F�_       �	�
��Yc�A�-*

losso�==�^�u       �	+���Yc�A�-*

lossR��<	*       �	�q��Yc�A�-*

loss6�m<'پ�       �	���Yc�A�-*

loss��.=�s�a       �	lˬ�Yc�A�-*

loss#��=u�r+       �	5{��Yc�A�-*

loss��-=����       �	�-��Yc�A�-*

losss|�=�v�       �	�߮�Yc�A�-*

loss �<vd��       �	?���Yc�A�-*

loss�	�<�	~        �	�L��Yc�A�-*

lossF�<�O;       �	���Yc�A�-*

loss/L�<��l       �	����Yc�A�-*

loss84<߿Og       �	�g��Yc�A�-*

loss�u�<�M�       �	2��Yc�A�-*

loss)�I=Z�G       �	�γ�Yc�A�-*

loss*�<�$`       �	���Yc�A�-*

loss�B=�h�m       �	�>��Yc�A�-*

loss�_�<��2       �	���Yc�A�-*

lossV�<
k�       �	ҫ��Yc�A�-*

lossu��=�P�1       �	�S��Yc�A�-*

loss���;�9�Q       �	���Yc�A�-*

lossd�<��3�       �	器�Yc�A�-*

loss�$�=��*(       �	�R��Yc�A�-*

lossfa"=�v8       �	����Yc�A�-*

loss#�=y��       �	B���Yc�A�-*

lossO�.<��	       �	�L��Yc�A�-*

loss_0�;�@��       �	���Yc�A�-*

loss?\<�	��       �	E���Yc�A�-*

loss�h�<���       �	)[��Yc�A�-*

lossx�<h��P       �	��Yc�A�-*

loss�5�==��       �	콾�Yc�A�-*

loss�Z�=���       �	Rd��Yc�A�-*

loss���<�%�       �	*��Yc�A�-*

loss1ky<t�       �	����Yc�A�-*

loss�16=&�Y�       �	�N��Yc�A�-*

loss�g<��?       �	����Yc�A�-*

loss4ȗ<Ҹ�       �	���Yc�A�-*

loss���;k���       �	�V��Yc�A�-*

loss@l*=����       �	 ���Yc�A�.*

loss��-<�o�	       �	#���Yc�A�.*

loss�B=�Đ       �	l���Yc�A�.*

loss|��<�/6       �	N|��Yc�A�.*

loss��f=W>�F       �	�!��Yc�A�.*

lossMl9< �S�       �	����Yc�A�.*

loss��=C�+       �	Sx��Yc�A�.*

lossX��<�k�       �	}#��Yc�A�.*

lossE��<�!�       �	"���Yc�A�.*

lossNh�<�$��       �	�h��Yc�A�.*

loss�y<t��       �	��Yc�A�.*

loss���<{4       �	$���Yc�A�.*

loss:a�=itU�       �	_���Yc�A�.*

loss��E=���t       �	�V��Yc�A�.*

loss^<���       �	A+��Yc�A�.*

lossr!Z<�       �	����Yc�A�.*

loss�H <�+v�       �	ͱ��Yc�A�.*

loss,�=���j       �	�g��Yc�A�.*

loss{5=�O�       �	�_��Yc�A�.*

loss���<]kO�       �	���Yc�A�.*

losszǨ<���       �	�!��Yc�A�.*

loss7��<�[i       �	\��Yc�A�.*

loss��=Ur�       �	����Yc�A�.*

loss�9M=(�?       �	�;��Yc�A�.*

loss:�f=��       �	����Yc�A�.*

loss��<5s�       �	���Yc�A�.*

lossA�<�
       �	�b��Yc�A�.*

loss��8=��kB       �	�	��Yc�A�.*

loss[��<=��       �	���Yc�A�.*

loss.4=z���       �	�p��Yc�A�.*

loss#�;<f�       �	&��Yc�A�.*

losste&=B%Z#       �	����Yc�A�.*

loss��=X���       �	c��Yc�A�.*

loss�s><xe}y       �	���Yc�A�.*

loss&&=����       �	����Yc�A�.*

loss�_;���       �	�X��Yc�A�.*

loss��&=�<g1       �	����Yc�A�.*

loss/=~A�       �	����Yc�A�.*

lossx��<ke��       �	�E��Yc�A�.*

loss�V�<�S�K       �	3j��Yc�A�.*

loss��;=��!�       �	N��Yc�A�.*

loss�o	=^,��       �	VI��Yc�A�.*

loss�JP=x��       �	����Yc�A�.*

loss���<��'f       �	y���Yc�A�.*

loss��=�f�       �	7���Yc�A�.*

loss�,�;�P�       �	%>��Yc�A�.*

loss@E<��%       �	V���Yc�A�.*

loss��9=b8       �	\���Yc�A�.*

loss!�;�S�       �	�U��Yc�A�.*

loss?<9���       �	����Yc�A�.*

loss���<䆥�       �	���Yc�A�.*

loss7�C<4�Ԅ       �	O��Yc�A�.*

loss�K=1��       �	V���Yc�A�.*

lossf�<��Ak       �	����Yc�A�.*

loss��4;��n�       �	Ҋ��Yc�A�.*

lossۮ�:��o       �	o.��Yc�A�.*

loss���;�_6       �	s���Yc�A�.*

lossH��<%���       �	9|��Yc�A�.*

loss���<����       �	�,��Yc�A�.*

loss���:��>�       �	����Yc�A�.*

loss�C<G�w       �	C���Yc�A�.*

loss&a=%I�       �	�1��Yc�A�.*

lossC�f;wJ�       �	���Yc�A�.*

loss�p<�=
       �		���Yc�A�.*

loss$"�<^��       �	1D��Yc�A�.*

loss�U5=l��       �	l���Yc�A�.*

loss��<'�b�       �	-���Yc�A�.*

lossl�	=�#�       �	�4��Yc�A�.*

loss=��<?�Ѭ       �	R���Yc�A�.*

loss(�!=�h*@       �	em��Yc�A�.*

losse=��       �	N&��Yc�A�.*

loss��= ]�T       �	I���Yc�A�.*

lossf��<��       �	ˆ��Yc�A�.*

loss�c==|�j�       �	�+��Yc�A�.*

loss8�	=`�&�       �	���Yc�A�.*

loss���<�Q�[       �	�s��Yc�A�.*

loss�S={�EQ       �	#��Yc�A�.*

loss,Er=����       �	"���Yc�A�.*

loss���<�yz�       �	�v��Yc�A�.*

lossW�q<BL�U       �	v��Yc�A�.*

loss@�4=,D�       �	l���Yc�A�.*

loss�o <��       �	�}��Yc�A�.*

loss�ɝ;�
�Z       �	�-��Yc�A�.*

loss�*c<x�_�       �	����Yc�A�.*

lossv�I;X��)       �	�� �Yc�A�.*

loss�L�;AG)       �	�:�Yc�A�.*

loss�<l<��       �	���Yc�A�.*

loss�f�<~g�I       �	[��Yc�A�.*

loss���<�إI       �	B�Yc�A�.*

loss�ރ<�I}       �	���Yc�A�.*

loss���<�w8�       �	$��Yc�A�.*

lossH�I=c�a       �	kD�Yc�A�.*

loss�^=~       �	[��Yc�A�.*

loss�W�=j�U�       �	q��Yc�A�.*

lossWz8<sd�       �	&7�Yc�A�.*

loss�3;CT�       �	���Yc�A�.*

loss&p=8Q�       �	��Yc�A�.*

loss�<���       �	�8	�Yc�A�.*

loss��;tC�r       �	H�	�Yc�A�.*

loss�j=	A4�       �	��Yc�A�.*

loss�yP=C��       �	B��Yc�A�.*

loss��M<5 ]�       �	t��Yc�A�.*

loss��5<�GT       �	NC�Yc�A�.*

loss��@=2;�       �	P��Yc�A�.*

loss��=_�4       �	���Yc�A�.*

loss�گ<$�+       �	�K�Yc�A�.*

loss�;G5X�       �	���Yc�A�.*

loss�2�<}�0B       �	�<�Yc�A�.*

loss�6=�)�C       �	���Yc�A�.*

loss��'=`d4       �		��Yc�A�.*

loss���<�S�w       �	=(�Yc�A�.*

loss��e;�	_�       �	u��Yc�A�.*

loss}�;��N�       �	py�Yc�A�.*

loss�A�=b�       �	�/�Yc�A�.*

loss�&=ݍ�g       �	(�0�Yc�A�.*

loss':�=/Z�       �	�]1�Yc�A�.*

loss���<���       �	��1�Yc�A�.*

lossc4=܆%T       �	q 3�Yc�A�.*

lossa�=�1u�       �	�4�Yc�A�.*

loss���<��g�       �	��4�Yc�A�.*

loss�M�<TXq�       �	�S5�Yc�A�.*

loss�Vf=�c��       �	�#6�Yc�A�.*

loss�3T={	��       �	��6�Yc�A�.*

loss���;Z�2�       �	l7�Yc�A�.*

loss��=�i�       �	8�Yc�A�.*

loss�k7=��/�       �	X�8�Yc�A�.*

lossF�=��       �	�D9�Yc�A�.*

loss�&S=,�5       �	��9�Yc�A�.*

loss�ٹ=��H        �	j�:�Yc�A�/*

lossSd<zm�       �	B#;�Yc�A�/*

loss�=����       �	��;�Yc�A�/*

loss�<6t�       �	2q<�Yc�A�/*

loss�S+=��oJ       �	�=�Yc�A�/*

loss��o<���       �	�=�Yc�A�/*

loss/P2=��S�       �	�]>�Yc�A�/*

losscq�;��U       �	O?�Yc�A�/*

loss =�Z�Q       �	@�?�Yc�A�/*

loss��<ț"       �	UQ@�Yc�A�/*

loss�B<g�]       �	��@�Yc�A�/*

loss�|B=�w5       �	k�A�Yc�A�/*

loss�@y;&8�(       �	�EB�Yc�A�/*

loss���<0�T       �	`�B�Yc�A�/*

loss\�a=-���       �	��C�Yc�A�/*

loss��=��J       �	�%D�Yc�A�/*

loss&qT=BPZC       �	`�D�Yc�A�/*

loss�gx<�/�\       �	mE�Yc�A�/*

loss���=�<o       �	RF�Yc�A�/*

loss �<�Z��       �	ƧF�Yc�A�/*

loss-/�<;Rh       �	FG�Yc�A�/*

loss��d<q��       �	T�G�Yc�A�/*

loss�<�"/       �	-�H�Yc�A�/*

loss[u�=(�       �	<I�Yc�A�/*

loss�Bv=�>D�       �	L�I�Yc�A�/*

loss���<���1       �	�yJ�Yc�A�/*

loss dt=?r!5       �	�K�Yc�A�/*

loss��B<[�x       �	��K�Yc�A�/*

loss�65=�5�z       �	��L�Yc�A�/*

losso�<(�å       �	�M�Yc�A�/*

loss:;�<!L�       �	z�N�Yc�A�/*

loss�8U<�@       �	bNO�Yc�A�/*

loss�X	=W���       �	��O�Yc�A�/*

lossV�q=͏�       �	Y�P�Yc�A�/*

lossM�:���H       �	EQ�Yc�A�/*

lossEԪ<�� O       �	��Q�Yc�A�/*

loss�=��*�       �	{�R�Yc�A�/*

loss�p�<]�       �	!S�Yc�A�/*

loss���=���G       �	��S�Yc�A�/*

lossQVr<��u       �	%]T�Yc�A�/*

loss���;�5�       �	0�T�Yc�A�/*

loss��z:#�ڸ       �	��U�Yc�A�/*

loss'�;/�       �	6:V�Yc�A�/*

loss��6<��L5       �	��V�Yc�A�/*

loss ��<���Y       �	zW�Yc�A�/*

loss��+=Ž�       �	�X�Yc�A�/*

loss#K<t5�       �	&�X�Yc�A�/*

lossla�:���       �	FY�Yc�A�/*

loss��=�6�p       �	��Y�Yc�A�/*

loss���<�!U�       �	�Z�Yc�A�/*

loss:�h;5[�m       �	�([�Yc�A�/*

loss�}=O��       �	�[�Yc�A�/*

loss��3=��%�       �	�x\�Yc�A�/*

loss�w =�g�h       �	5']�Yc�A�/*

losss1"="坔       �	L�]�Yc�A�/*

loss�=��X       �	u�^�Yc�A�/*

loss�q�<,���       �	I._�Yc�A�/*

loss*��<?��_       �	�O`�Yc�A�/*

loss��<�*�|       �	��`�Yc�A�/*

loss�{[<*%       �	��a�Yc�A�/*

loss[.=�jow       �	�>b�Yc�A�/*

loss���;���0       �	��b�Yc�A�/*

lossAə=;6��       �	�c�Yc�A�/*

loss�h1<$�i=       �	�"d�Yc�A�/*

loss�a�<�-U�       �	��d�Yc�A�/*

loss&�<a�q]       �	�e�Yc�A�/*

lossO8=�-@�       �	�Sg�Yc�A�/*

losst�<��T�       �	Byh�Yc�A�/*

loss;�><QQ       �	��j�Yc�A�/*

lossE�<��       �	W'k�Yc�A�/*

loss^�<����       �	�k�Yc�A�/*

loss���<E�1�       �	�l�Yc�A�/*

loss���<�'�       �	�Em�Yc�A�/*

lossd/=B       �	��m�Yc�A�/*

loss�=*]��       �	��n�Yc�A�/*

loss�a�<�_�=       �	/4o�Yc�A�/*

lossg=^��W       �	e�o�Yc�A�/*

loss�`�<���       �	b�p�Yc�A�/*

lossÛ<�yo_       �	"q�Yc�A�/*

loss�H=o�k       �	��q�Yc�A�/*

loss!s4<�ǋ       �	Ur�Yc�A�/*

lossT��<��'�       �	��r�Yc�A�/*

loss��<hza�       �	)�s�Yc�A�/*

lossE��<����       �	�;t�Yc�A�/*

lossR(�<�+�	       �	��t�Yc�A�/*

lossD	�<�        �	�vu�Yc�A�/*

loss�_<N�ќ       �	v�Yc�A�/*

loss�=��Tw       �	�v�Yc�A�/*

lossk�<-Y3>       �	�Uw�Yc�A�/*

loss@��<j+��       �	��w�Yc�A�/*

loss祿=�,�4       �	��x�Yc�A�/*

loss*OO<��;/       �	)#y�Yc�A�/*

loss���<���Y       �	��y�Yc�A�/*

loss��=+
H       �	iRz�Yc�A�/*

lossA�<�,�`       �	��z�Yc�A�/*

loss��=���5       �	ߧ{�Yc�A�/*

loss���<x���       �	�=|�Yc�A�/*

loss��j=�uo       �	��|�Yc�A�/*

loss�hw;>3��       �	�}�Yc�A�/*

loss�U�<WYf�       �	;q~�Yc�A�/*

loss�b�;o,��       �	��Yc�A�/*

loss*��<�� B       �	X��Yc�A�/*

loss{*=z�C       �	�E��Yc�A�/*

loss�I�<H�2�       �	���Yc�A�/*

loss�3'=.2��       �	D���Yc�A�/*

loss݅�<ຂq       �	�>��Yc�A�/*

loss�Di=S��       �	gՂ�Yc�A�/*

loss=@F��       �	7l��Yc�A�/*

loss#�=���c       �	���Yc�A�/*

lossǥ<i��H       �	=���Yc�A�/*

loss_��< �C�       �	�<��Yc�A�/*

loss�Y='��       �	J҅�Yc�A�/*

loss��<h��       �	Ou��Yc�A�/*

loss&4=s	�]       �	��Yc�A�/*

loss��G=��;       �	����Yc�A�/*

loss;+=����       �	�L��Yc�A�/*

loss�==�G7c       �	e��Yc�A�/*

loss=�<.��       �	B|��Yc�A�/*

loss@�w<�g       �	���Yc�A�/*

loss*�9=�u�       �	ѱ��Yc�A�/*

lossV��<W!�       �	�M��Yc�A�/*

loss_��<w��2       �	:��Yc�A�/*

loss%b%=���       �	͐��Yc�A�/*

lossH�x<'�U	       �	�ڍ�Yc�A�/*

lossH�z<�v_       �	�y��Yc�A�/*

loss���;t���       �	#��Yc�A�/*

loss@�<NF��       �	���Yc�A�/*

loss�==�Ԯr       �	B��Yc�A�/*

loss��=d��       �	�Ԑ�Yc�A�0*

loss�F=4�(       �	�m��Yc�A�0*

loss�=�!��       �	W��Yc�A�0*

loss��6=M"�]       �	l��Yc�A�0*

loss���;"-nm       �	���Yc�A�0*

loss��=໱�       �	8��Yc�A�0*

lossS=��7�       �	�Ҕ�Yc�A�0*

lossGR=F�̥       �	n��Yc�A�0*

loss���=@���       �	Z��Yc�A�0*

loss��Y=C-�}       �	����Yc�A�0*

loss�J�<#0�       �	�H��Yc�A�0*

loss�!=���       �	X��Yc�A�0*

loss�C�;����       �	>���Yc�A�0*

lossN:�<��       �	�2��Yc�A�0*

loss(�e=�[�X       �	�ə�Yc�A�0*

lossM�.=����       �	�a��Yc�A�0*

loss��=��,       �	����Yc�A�0*

loss�=9�s;       �	є��Yc�A�0*

lossI�4<T$       �	]n��Yc�A�0*

loss[!�<Jq3       �	m��Yc�A�0*

loss�t�;��K�       �	(���Yc�A�0*

loss�c�<+X4       �	
1��Yc�A�0*

loss��e<��%�       �	�՞�Yc�A�0*

loss�W�;����       �	l��Yc�A�0*

lossT�==?k��       �	A��Yc�A�0*

loss,.<M@�       �	����Yc�A�0*

loss3X�<��!       �	�<��Yc�A�0*

lossWXG= �O�       �	$ӡ�Yc�A�0*

lossD�Q<�ʔ\       �	�l��Yc�A�0*

loss �;<w F	       �	���Yc�A�0*

loss*�=M��"       �	V���Yc�A�0*

loss �;>�)       �	be��Yc�A�0*

lossv��<���       �	���Yc�A�0*

loss�O�=QO�       �	t���Yc�A�0*

lossd�-=V��u       �	�]��Yc�A�0*

loss?=\e�       �	����Yc�A�0*

lossR2<CQ��       �	|���Yc�A�0*

loss��;�O�4       �	yZ��Yc�A�0*

loss�R�=v��&       �	���Yc�A�0*

loss��;��       �	���Yc�A�0*

loss���;�0��       �	���Yc�A�0*

lossm(�<6f��       �	*���Yc�A�0*

loss_W<�4J�       �	�'��Yc�A�0*

lossd^=�7\       �	,���Yc�A�0*

lossT�<S��,       �	�Q��Yc�A�0*

loss���<$�j       �	���Yc�A�0*

loss�r=���       �	{���Yc�A�0*

lossC��<c F9       �	+��Yc�A�0*

loss-�=�.�b       �	�į�Yc�A�0*

loss���;�`       �	�d��Yc�A�0*

loss�x�<�K :       �	$��Yc�A�0*

loss�Y<����       �	,���Yc�A�0*

loss�T�<�5��       �	/3��Yc�A�0*

losss=Z�8       �	.ʲ�Yc�A�0*

loss�TV=��;       �	u���Yc�A�0*

lossH=��W       �	<��Yc�A�0*

loss�\\<�֘       �	Ҵ�Yc�A�0*

loss�=�W�L       �	{g��Yc�A�0*

loss��R<��j       �	:#��Yc�A�0*

loss��<x�       �	����Yc�A�0*

loss�u�<ag=�       �	����Yc�A�0*

lossW�=���J       �	�'��Yc�A�0*

loss���<?�\       �	�ĸ�Yc�A�0*

loss���<N�Ll       �	i��Yc�A�0*

loss�]=��0&       �	���Yc�A�0*

loss���<ՆR�       �	f���Yc�A�0*

loss�)�<6��       �	�8��Yc�A�0*

loss�ɀ<�<��       �	�ϻ�Yc�A�0*

loss�	=IP�       �	�o��Yc�A�0*

loss�=Ldݺ       �	5��Yc�A�0*

loss}�<Y2}�       �	⯽�Yc�A�0*

lossJʇ=q�I       �	aS��Yc�A�0*

loss���=�x)�       �	4��Yc�A�0*

loss�=���       �	����Yc�A�0*

loss�2�=��Q�       �	�F��Yc�A�0*

loss�!a=Z��V       �	����Yc�A�0*

lossJa=��oZ       �	�t��Yc�A�0*

loss3� <��54       �	���Yc�A�0*

lossjD<H�F       �	ʥ��Yc�A�0*

loss+�=���       �	>��Yc�A�0*

lossnG=���]       �	����Yc�A�0*

loss�xH=O3q       �	���Yc�A�0*

loss-w==���       �	g'��Yc�A�0*

loss_o%=��       �	����Yc�A�0*

loss��N<�A       �	�h��Yc�A�0*

lossR��<&-�       �	
��Yc�A�0*

lossO�h<A�^8       �	(���Yc�A�0*

loss쀛=(��N       �	6Z��Yc�A�0*

lossN�
<�D�       �	���Yc�A�0*

lossf�e<�y�       �	6���Yc�A�0*

loss���<�P��       �	�T��Yc�A�0*

loss�F=��V?       �	����Yc�A�0*

loss���=�1       �	���Yc�A�0*

loss��<O#\�       �	����Yc�A�0*

loss��>ĞI�       �	����Yc�A�0*

loss(�D=f���       �	����Yc�A�0*

loss4�*<���       �	����Yc�A�0*

loss��<@#�       �	ߊ��Yc�A�0*

loss���<�o��       �	nM��Yc�A�0*

lossm��<U
Q       �	���Yc�A�0*

loss�2�<o�R�       �	�J��Yc�A�0*

loss�ը<̙E�       �	�;��Yc�A�0*

loss�p�<��!�       �	���Yc�A�0*

loss�b<�?m       �	����Yc�A�0*

lossן�<W{qt       �	
���Yc�A�0*

loss���<\y�+       �	Tn��Yc�A�0*

loss� =���k       �	̲��Yc�A�0*

loss��<!L��       �	����Yc�A�0*

lossZz�<Z���       �	*��Yc�A�0*

losst��<����       �	ػ��Yc�A�0*

lossE��<GB8B       �	b��Yc�A�0*

lossZY;H�y�       �	7 ��Yc�A�0*

loss��7<�x�       �	H���Yc�A�0*

loss��y</�\V       �	%<��Yc�A�0*

lossc y<N �7       �	����Yc�A�0*

loss<�O<�1˳       �	�u��Yc�A�0*

lossC\	=|h�0       �	���Yc�A�0*

losss;8=��       �	����Yc�A�0*

loss�?2=
7U�       �	~R��Yc�A�0*

loss��)=lK�b       �	����Yc�A�0*

loss�~=��c�       �	����Yc�A�0*

loss��I=rK�:       �	�,��Yc�A�0*

loss�g�;��y       �	����Yc�A�0*

loss솁=� ��       �	���Yc�A�0*

loss#$(=n0E       �	*��Yc�A�0*

loss�n=]���       �	a���Yc�A�0*

loss��B=\��       �	�]��Yc�A�0*

loss���<b�M�       �	o���Yc�A�0*

loss3[�<.�U       �	/���Yc�A�1*

loss�U={d�H       �	�j��Yc�A�1*

loss�6�<B�"�       �	���Yc�A�1*

loss��a=мS�       �	���Yc�A�1*

loss;�=��'�       �	=��Yc�A�1*

loss�N�<u/�2       �	x���Yc�A�1*

loss��<�VQ�       �	�l��Yc�A�1*

lossE[4=$?a�       �	���Yc�A�1*

loss��g=e:�       �	���Yc�A�1*

loss��s;�~�       �	�?��Yc�A�1*

loss��<���W       �	���Yc�A�1*

loss ��<L��       �	�{��Yc�A�1*

loss{0n<]p�       �	I��Yc�A�1*

loss�g3=2�K�       �	��Yc�A�1*

loss�i�<+AV�       �	#���Yc�A�1*

loss���<��1�       �	�E��Yc�A�1*

lossA�=-��       �	����Yc�A�1*

loss��<R��       �	���Yc�A�1*

loss;c:<���a       �	.���Yc�A�1*

loss/I;����       �	����Yc�A�1*

loss��=���[       �	=��Yc�A�1*

lossƞM<,X�       �	����Yc�A�1*

lossS߿;�8�       �	7n��Yc�A�1*

loss�]Z<�Dt       �	���Yc�A�1*

loss�� =n�QS       �	,���Yc�A�1*

loss)&�<����       �	Q��Yc�A�1*

loss!R=X��\       �	 ���Yc�A�1*

loss��S=s'��       �	����Yc�A�1*

loss��<=�&�       �	�^��Yc�A�1*

loss!��<�,)�       �	#���Yc�A�1*

loss�3=Ǭ`�       �	8���Yc�A�1*

loss\�f<�'��       �	�>��Yc�A�1*

loss�3�=��5�       �	����Yc�A�1*

loss]�=���       �	|��Yc�A�1*

loss��=��W       �	���Yc�A�1*

loss�b	=��0}       �	'���Yc�A�1*

loss�I�<S��'       �	 | �Yc�A�1*

loss,Eh<`�:f       �	�Yc�A�1*

loss�K>=�Ò�       �	ͭ�Yc�A�1*

lossox_<A�x       �	�R�Yc�A�1*

loss���<�'��       �	��Yc�A�1*

loss���<Ć��       �	ۧ�Yc�A�1*

loss��<�wr�       �	�x�Yc�A�1*

loss��<��-       �	�Yc�A�1*

loss�"=�m�       �	���Yc�A�1*

losss��<�D�T       �	�[�Yc�A�1*

loss�D�<���       �	� �Yc�A�1*

loss.�=kr��       �	��Yc�A�1*

loss�
�<!�ؒ       �	�:�Yc�A�1*

lossd_=?��       �	[��Yc�A�1*

loss���<U��       �	�l	�Yc�A�1*

loss
I=���|       �	B
�Yc�A�1*

lossl�<8#_�       �	w�
�Yc�A�1*

lossi9=�i?y       �	�?�Yc�A�1*

loss$�W=%�>       �	R��Yc�A�1*

loss�+=���       �	��Yc�A�1*

loss�N�<�~�       �	���Yc�A�1*

lossŖ{=ӛg�       �	?��Yc�A�1*

loss*�0=�0       �	�9�Yc�A�1*

loss�bK<W~��       �	]6�Yc�A�1*

loss�'�<v�Z�       �	}?�Yc�A�1*

lossN}5<��67       �	#0�Yc�A�1*

loss$4�=�V��       �	�c�Yc�A�1*

loss��c=fҴ�       �	�Yc�A�1*

lossX��<b��       �	���Yc�A�1*

loss�U=�iK�       �	���Yc�A�1*

loss���<-��
       �	f�Yc�A�1*

loss���<n�       �	p$�Yc�A�1*

loss�j�;j�%       �	j��Yc�A�1*

loss ML<|U��       �	px�Yc�A�1*

loss�a�;ح�r       �	Q�Yc�A�1*

lossC�_<8�|�       �	��Yc�A�1*

loss�}<b��E       �	jP�Yc�A�1*

loss_�=��2"       �	���Yc�A�1*

loss�=\%�       �	���Yc�A�1*

lossI�<Lm��       �	-�Yc�A�1*

loss�Y9=f(W       �	���Yc�A�1*

lossٵ<���       �	�[�Yc�A�1*

lossv�g=󍴣       �	��Yc�A�1*

lossN�<J���       �	R��Yc�A�1*

lossj�<-        �	�9�Yc�A�1*

lossL�W<�I�v       �	���Yc�A�1*

lossq�;H((       �	�� �Yc�A�1*

lossҮ�<-۳       �	(!�Yc�A�1*

lossC��;�a��       �	��!�Yc�A�1*

lossdb(<���       �	�s"�Yc�A�1*

loss}��<�q�       �	�#�Yc�A�1*

lossq�=NQQ       �	�#�Yc�A�1*

loss�Cz<2H�>       �	�Q$�Yc�A�1*

loss��<��ْ       �	/l%�Yc�A�1*

lossw(<��l�       �	7&�Yc�A�1*

loss�+.=���       �	�&�Yc�A�1*

loss���=
e�,       �	�c'�Yc�A�1*

loss$<�=��,�       �	(�Yc�A�1*

loss-�?= �z�       �	=�(�Yc�A�1*

loss���<��%�       �	�Z)�Yc�A�1*

loss*�<k��       �	��)�Yc�A�1*

lossl�5<v/�       �	��*�Yc�A�1*

loss&Y"<[V�       �	BC+�Yc�A�1*

loss!g�<C�s       �	*�+�Yc�A�1*

loss�F�;�>rz       �	��,�Yc�A�1*

loss\�=L+�       �	� -�Yc�A�1*

loss��=���z       �	��-�Yc�A�1*

lossl9=��2�       �	�j.�Yc�A�1*

loss�6�=`_�k       �	8/�Yc�A�1*

loss��<���       �	�/�Yc�A�1*

loss@�h<8��       �	��0�Yc�A�1*

loss�H�<���J       �	*91�Yc�A�1*

loss�A=΄��       �	g�1�Yc�A�1*

lossI�<g@�       �	�r2�Yc�A�1*

loss/��<�F
       �	E3�Yc�A�1*

loss�2�<�D       �	�3�Yc�A�1*

loss}��<Nb'�       �	F4�Yc�A�1*

loss��;��=�       �	��4�Yc�A�1*

loss�{�;y��p       �	�}5�Yc�A�1*

loss{�'=�`*	       �	P6�Yc�A�1*

loss?��<���h       �	l�6�Yc�A�1*

loss�<�o*O       �	v�7�Yc�A�1*

loss�X<�gA       �	&8�Yc�A�1*

loss��u<8soP       �	��8�Yc�A�1*

lossGc=��B       �	2Y9�Yc�A�1*

loss.Ֆ<8g
       �	��9�Yc�A�1*

loss��I=1Z       �	��:�Yc�A�1*

loss!=^nd       �	�5;�Yc�A�1*

lossz��=INs�       �	��;�Yc�A�1*

lossTv�;rr�       �	�<�Yc�A�1*

loss�ֺ<6J
	       �	e=�Yc�A�1*

loss�(=�⬌       �	N>�Yc�A�1*

loss4��<IP]R       �	��>�Yc�A�2*

loss�<��x       �	ގ?�Yc�A�2*

loss�=ͨ��       �	]7@�Yc�A�2*

lossʹ�<g���       �	^�@�Yc�A�2*

loss�zF<�v/�       �	?�A�Yc�A�2*

lossZD<�s       �	&�B�Yc�A�2*

lossl�5=n*��       �	�.C�Yc�A�2*

lossZ�>=#�]2       �	��C�Yc�A�2*

loss�j'=<C/�       �	�jD�Yc�A�2*

loss���<�b       �	�E�Yc�A�2*

lossyw=B��       �	ΧE�Yc�A�2*

loss�=��e       �	�BF�Yc�A�2*

loss��#=�*<       �	�F�Yc�A�2*

loss��=���%       �	�G�Yc�A�2*

loss��=�5�       �	�8H�Yc�A�2*

lossFi=%Ƌ       �	K�H�Yc�A�2*

loss�=V<,�       �	׊I�Yc�A�2*

loss�hr;��5       �	�*J�Yc�A�2*

lossd&<Lr��       �	��J�Yc�A�2*

loss$p�<zoK�       �	hK�Yc�A�2*

loss�i=2��       �	7�K�Yc�A�2*

loss�`�<�S�       �	p�L�Yc�A�2*

loss*&�<t�lM       �	�`M�Yc�A�2*

lossؗ�;���       �	?UN�Yc�A�2*

lossé^={���       �	T�N�Yc�A�2*

losslĈ=���       �	W�O�Yc�A�2*

loss�a=fm�       �	��P�Yc�A�2*

loss�<)�Q       �	:Q�Yc�A�2*

loss/IE=�w�       �	� R�Yc�A�2*

lossZ��<DBn�       �	��R�Yc�A�2*

lossfk=ŝ��       �	��S�Yc�A�2*

loss�b=�nF�       �	�`T�Yc�A�2*

lossW��<��,�       �	\U�Yc�A�2*

loss{�=�s       �	��U�Yc�A�2*

lossst�< 'B       �	�5V�Yc�A�2*

loss��I=$_��       �	J
W�Yc�A�2*

lossm�`=����       �	��W�Yc�A�2*

loss�u<�&�       �	�<X�Yc�A�2*

loss���<��OO       �	J�X�Yc�A�2*

loss�"�<Y��Q       �	LnY�Yc�A�2*

loss�{�<B&�4       �	�Z�Yc�A�2*

loss��O=�2�       �	j�Z�Yc�A�2*

losst�=$�7       �	�[�Yc�A�2*

loss�W�<}ܯ�       �	\�Yc�A�2*

loss�� =yy�       �	̳\�Yc�A�2*

loss}�;��m�       �	{O]�Yc�A�2*

loss��u=UXT       �	^�Yc�A�2*

lossW��=m���       �	 �^�Yc�A�2*

loss]`=��ǟ       �	�F_�Yc�A�2*

lossaX
=�77�       �	�_�Yc�A�2*

lossR�={��W       �	-�`�Yc�A�2*

loss���=8us�       �	�5a�Yc�A�2*

loss}Xs<k�޿       �	V�a�Yc�A�2*

loss�`�<�6>       �	wb�Yc�A�2*

loss��<7m�       �	�c�Yc�A�2*

lossNd=no�       �	��c�Yc�A�2*

loss]�1=L��<       �	�Qd�Yc�A�2*

lossQ�<��        �	��d�Yc�A�2*

loss�g=��,       �	��e�Yc�A�2*

loss`,�<����       �	�}f�Yc�A�2*

loss��<����       �	mg�Yc�A�2*

losso��<��z       �	e�g�Yc�A�2*

loss�8�;~hB       �	�\h�Yc�A�2*

lossVe�;��`Z       �	&�h�Yc�A�2*

loss�a
=r�8       �	�i�Yc�A�2*

loss�y=��[n       �	9j�Yc�A�2*

loss�t�<P0��       �	�j�Yc�A�2*

lossq�5<��(       �	nk�Yc�A�2*

loss�>G<�0c�       �	+l�Yc�A�2*

loss��^<�{�       �	��l�Yc�A�2*

loss1v-<��        �	�`m�Yc�A�2*

loss:��<Nu��       �	�	n�Yc�A�2*

lossvp�<_YU       �	�n�Yc�A�2*

loss�d<k��       �	�=o�Yc�A�2*

loss��=��       �	��o�Yc�A�2*

lossj©<]d�       �	lxp�Yc�A�2*

loss&��<m�z       �	8q�Yc�A�2*

losso�C<���R       �	�q�Yc�A�2*

lossL�=���       �	4�r�Yc�A�2*

lossN/[=��:�       �	6s�Yc�A�2*

lossJ�B=�8�V       �	��s�Yc�A�2*

lossq��=ͮ��       �	�Rt�Yc�A�2*

lossR�=;ڢi       �	��t�Yc�A�2*

losssj�<͇=�       �	��u�Yc�A�2*

loss�M$=�1<       �	`=v�Yc�A�2*

loss-#=�A       �	��v�Yc�A�2*

losss�<� �w       �	|w�Yc�A�2*

loss���;Z��       �	)%x�Yc�A�2*

lossf��<����       �	w�x�Yc�A�2*

lossOJs<=�^       �	��y�Yc�A�2*

loss��<�5�       �	�Vz�Yc�A�2*

loss�@�<s&+l       �	�{�Yc�A�2*

lossf��<��{       �	��{�Yc�A�2*

loss��b<}Z�/       �	��|�Yc�A�2*

loss�T:<$Hu       �	�A}�Yc�A�2*

loss�	=���       �	��}�Yc�A�2*

lossq��<�       �	��~�Yc�A�2*

loss��<]�L       �	��Yc�A�2*

loss|%�<����       �	M��Yc�A�2*

losst_�<v��       �	�V��Yc�A�2*

loss�z�<)p�M       �	o��Yc�A�2*

loss�:=LN       �	����Yc�A�2*

loss#<��R       �	%��Yc�A�2*

lossj�<�.#r       �	����Yc�A�2*

loss�f�;�Ɓ�       �	T��Yc�A�2*

loss��=����       �	����Yc�A�2*

lossz99<�i�       �	o���Yc�A�2*

lossNYF=���       �	jK��Yc�A�2*

loss�-E<7�0�       �	���Yc�A�2*

lossB�<ڮ��       �	����Yc�A�2*

lossT`=d�K       �	� ��Yc�A�2*

loss��<�X�       �	U���Yc�A�2*

loss uZ=���F       �	�X��Yc�A�2*

loss4�;�ZO�       �	C ��Yc�A�2*

loss
s�<)��       �	,���Yc�A�2*

loss�b�<�ʌ       �	P9��Yc�A�2*

loss��-=��4e       �	~��Yc�A�2*

loss���<�qԇ       �	�~��Yc�A�2*

loss�w-=���       �	���Yc�A�2*

lossJ�(<_�}       �	>>��Yc�A�2*

lossl��<���       �	�z��Yc�A�2*

loss�<*�{       �	 ��Yc�A�2*

loss\+�;7���       �	SΏ�Yc�A�2*

loss	;wX}�       �	p��Yc�A�2*

lossp�=G�       �	����Yc�A�2*

loss��=\��        �	�`��Yc�A�2*

loss��=;�]       �	���Yc�A�2*

loss[7=EZF�       �	Zd��Yc�A�2*

loss���<2��       �	����Yc�A�3*

loss�Ѣ=�]'H       �	�B��Yc�A�3*

loss~�!<��?�       �	�3��Yc�A�3*

loss�?W<3���       �	���Yc�A�3*

loss�o=&��       �	׿��Yc�A�3*

lossZ�<�F��       �	����Yc�A�3*

loss/��:nx�K       �	����Yc�A�3*

lossc��<�c_�       �	�U��Yc�A�3*

lossL�<���       �	����Yc�A�3*

lossC+�;%`       �	ƣ��Yc�A�3*

losshCH<_[M.       �	�ȝ�Yc�A�3*

lossM�;�Fk�       �	�~��Yc�A�3*

loss$,�<C:�       �	32��Yc�A�3*

loss��<��ˀ       �	�g��Yc�A�3*

lossEZp:����       �	���Yc�A�3*

loss�u�:�ow       �	O��Yc�A�3*

loss�F�;�д       �	o��Yc�A�3*

loss[" <�       �	��Yc�A�3*

lossF	e<�MCp       �	����Yc�A�3*

loss�:���       �	+3��Yc�A�3*

loss�a=m�&R       �	�Х�Yc�A�3*

loss\��=�(j        �	3k��Yc�A�3*

loss�B;��       �	a��Yc�A�3*

loss���<;̆       �	幧�Yc�A�3*

lossD�7=2�6A       �	�b��Yc�A�3*

loss�g=g��       �	����Yc�A�3*

loss|�<8       �	 ���Yc�A�3*

loss�-�<���Q       �	5��Yc�A�3*

loss��m=��*       �	�Ϫ�Yc�A�3*

lossaȷ<,%�0       �	x��Yc�A�3*

loss!t(=�'�       �	z��Yc�A�3*

loss�'=��s       �	o���Yc�A�3*

loss�=Ô�       �	τ��Yc�A�3*

loss��6<h*       �	}#��Yc�A�3*

loss�q�<;���       �	����Yc�A�3*

lossZ��<G`!,       �	*R��Yc�A�3*

loss >�<y�I       �	��Yc�A�3*

lossRl�<���       �	Ό��Yc�A�3*

loss���;�7�       �	�U��Yc�A�3*

loss�a<k)lO       �	N��Yc�A�3*

loss=I�<���W       �	����Yc�A�3*

loss���;���g       �	> ��Yc�A�3*

loss�[�;A��       �	����Yc�A�3*

loss)0V<�g)�       �	�N��Yc�A�3*

loss׿l<\�5       �	*��Yc�A�3*

loss�؉;��u�       �	���Yc�A�3*

loss��!=��8       �	��Yc�A�3*

losst�;d���       �	ܺ��Yc�A�3*

loss�j�<t�	       �	~U��Yc�A�3*

loss�L�<a�n�       �	���Yc�A�3*

lossq{;=�]0       �	�Yc�A�3*

loss!�=Jy|       �	�%��Yc�A�3*

loss@��;*���       �	����Yc�A�3*

lossO�=��y       �	�T��Yc�A�3*

loss�&<(@��       �	���Yc�A�3*

loss�z<��0J       �	���Yc�A�3*

loss���<2�u�       �	n��Yc�A�3*

loss�ϩ;�Z�       �	t��Yc�A�3*

loss��j<�!~�       �	A���Yc�A�3*

loss�jb=$u       �	k,��Yc�A�3*

loss3�=}T}       �	ܾ�Yc�A�3*

lossV"�<)��\       �	\u��Yc�A�3*

lossZ9�;C�O�       �	7��Yc�A�3*

loss�}p<��q       �	���Yc�A�3*

losso�|<���       �	�`��Yc�A�3*

loss��<Z���       �	���Yc�A�3*

losslK�<��}y       �	 ���Yc�A�3*

lossIg<��qU       �	�F��Yc�A�3*

loss��=1�7#       �	����Yc�A�3*

lossD�:Yh��       �	F}��Yc�A�3*

loss��<��T       �	���Yc�A�3*

loss�o;���"       �	���Yc�A�3*

lossXM,<�c�$       �	,E��Yc�A�3*

lossd�y<�G�       �	����Yc�A�3*

loss&��<��&-       �	ƈ��Yc�A�3*

loss��e=���       �	���Yc�A�3*

loss�b�<��k"       �	F���Yc�A�3*

loss��P=�_$       �	�X��Yc�A�3*

loss���<�'       �	N���Yc�A�3*

lossVAI<��u       �	����Yc�A�3*

loss���<0��	       �	�b��Yc�A�3*

loss��=R�W       �	� ��Yc�A�3*

loss��=�a�C       �	���Yc�A�3*

lossh�';�u��       �	�C��Yc�A�3*

lossi!E=��G       �	����Yc�A�3*

loss��%=ԗ(       �	ۉ��Yc�A�3*

loss@�'=�~�f       �	�]��Yc�A�3*

lossm�=7�q       �	����Yc�A�3*

lossx�=��       �	؝��Yc�A�3*

loss�:r;��ͨ       �	BA��Yc�A�3*

loss�=<=�VU       �	Q���Yc�A�3*

lossb�<1       �	�}��Yc�A�3*

loss�`�=tL�       �	��Yc�A�3*

lossR�k<���       �	���Yc�A�3*

loss�>9=� �       �	�C��Yc�A�3*

loss���<8�
7       �	O���Yc�A�3*

loss�)�=�i��       �	,���Yc�A�3*

loss�F4<e�
K       �	�#��Yc�A�3*

loss��<���       �	Z���Yc�A�3*

loss�p=1�[�       �	N��Yc�A�3*

loss_"�;��6�       �	����Yc�A�3*

loss��=g�X^       �	�|��Yc�A�3*

loss���<��~�       �	f��Yc�A�3*

lossx�U=Fi�       �	����Yc�A�3*

lossD�<���#       �	�[��Yc�A�3*

lossX�<�.��       �	����Yc�A�3*

loss*\&=��F�       �	���Yc�A�3*

loss�C=���       �	�/��Yc�A�3*

loss��9=/k��       �	����Yc�A�3*

loss4+�;֏�y       �	!Z��Yc�A�3*

loss���<��?       �	����Yc�A�3*

lossDܐ=�CoN       �	����Yc�A�3*

loss��<E�n�       �	='��Yc�A�3*

loss�d�<��^       �	���Yc�A�3*

loss#�&=`��s       �	Rb��Yc�A�3*

loss/*<،C       �	����Yc�A�3*

loss�+'=���       �	���Yc�A�3*

loss���<�&�       �	�6��Yc�A�3*

loss�
�<�C�       �	V���Yc�A�3*

lossŇ�<��\       �	�m��Yc�A�3*

loss&B.=3\Z�       �	���Yc�A�3*

lossRš<��       �	Y���Yc�A�3*

lossب�:��~       �	V��Yc�A�3*

loss��;�|�        �	A���Yc�A�3*

loss���<��B       �	G� �Yc�A�3*

loss�`<X_t!       �	�+�Yc�A�3*

loss#Q�=����       �	*��Yc�A�3*

loss]N=�0n�       �	a�Yc�A�3*

loss�i�;����       �	i��Yc�A�4*

lossW3�:O��       �	ٓ�Yc�A�4*

lossa�=;c�e~       �	�.�Yc�A�4*

loss���=iJ�=       �	���Yc�A�4*

loss�p�<5��O       �	�g�Yc�A�4*

loss&�=)�4�       �	���Yc�A�4*

loss�=�$��       �	��Yc�A�4*

loss��;�8ʌ       �	�;�Yc�A�4*

loss�#=�ĩ       �	���Yc�A�4*

loss.K�<]��       �	���Yc�A�4*

loss��=<_NA�       �	�%	�Yc�A�4*

loss�Ѵ<���       �	`�	�Yc�A�4*

lossO�L<�Ym       �	d
�Yc�A�4*

loss�}=o�       �	��Yc�A�4*

lossȚ�=@�{�       �	]��Yc�A�4*

loss�o�<wE^       �	G>�Yc�A�4*

lossDK"=�9�       �	�T�Yc�A�4*

loss���<�       �	��Yc�A�4*

loss��<��>f       �	��Yc�A�4*

loss�oK<v���       �	�n�Yc�A�4*

lossێ<�       �	���Yc�A�4*

loss��;��       �	b��Yc�A�4*

loss��=��PW       �	~�Yc�A�4*

loss��;gnzG       �	<��Yc�A�4*

loss� -<fK%       �	-`�Yc�A�4*

loss��'=� ��       �	���Yc�A�4*

loss��<��~�       �	���Yc�A�4*

lossI�<U�َ       �	k*�Yc�A�4*

loss�g;�Z\�       �	��Yc�A�4*

loss��< e��       �	"��Yc�A�4*

loss��<���       �	Af�Yc�A�4*

loss-l=ۆ�7       �	q�Yc�A�4*

loss��'=��$       �	��Yc�A�4*

lossX|�<��`�       �	"n�Yc�A�4*

loss2#�<�i�^       �	��Yc�A�4*

loss��<���a       �	N��Yc�A�4*

loss��;=��,       �	�7�Yc�A�4*

loss{�<?��       �	=��Yc�A�4*

losst�<5��       �	�i�Yc�A�4*

loss��=�s-�       �	e �Yc�A�4*

loss�}�;�s       �	_��Yc�A�4*

loss��=P&E@       �	;7�Yc�A�4*

loss�t�<o���       �	���Yc�A�4*

loss<��?�       �	�i �Yc�A�4*

lossƪ�<Բ1.       �	�!�Yc�A�4*

loss�%�;\�Dg       �	��!�Yc�A�4*

loss�L(<��       �	VH"�Yc�A�4*

loss��?=3�>�       �	��"�Yc�A�4*

loss�_d<�R�k       �	�v#�Yc�A�4*

lossiw�<��;�       �	$�Yc�A�4*

loss�8�=GFd�       �	?�$�Yc�A�4*

loss
�<����       �	��%�Yc�A�4*

loss	��<*�\b       �	2r&�Yc�A�4*

loss<Ԏ=�ݹ�       �	
'�Yc�A�4*

loss�u}<㈶F       �	��'�Yc�A�4*

loss��7=U�Q       �	~U(�Yc�A�4*

loss�k�<,��h       �	ڌ)�Yc�A�4*

losss�,=q�       �	�&*�Yc�A�4*

loss��;+�\       �	+�*�Yc�A�4*

lossā.=���       �	
�+�Yc�A�4*

loss>=���       �	x&,�Yc�A�4*

loss!��< ���       �	A�,�Yc�A�4*

loss(�,=���       �	O�-�Yc�A�4*

loss �_<
L%�       �	�1.�Yc�A�4*

loss�<<g�dD       �	��.�Yc�A�4*

lossJn�<౒�       �	0e/�Yc�A�4*

loss���=�/�I       �	S0�Yc�A�4*

lossv,X=S�       �	8�0�Yc�A�4*

loss�� =���H       �	61�Yc�A�4*

loss\]�;�       �	��1�Yc�A�4*

lossz�S=�I�       �	Hp2�Yc�A�4*

lossM'�<q���       �	3�Yc�A�4*

loss	k@<�k�3       �	��3�Yc�A�4*

loss���<V���       �	�V4�Yc�A�4*

loss�Y�<��U�       �	� 5�Yc�A�4*

loss%F�<��'�       �	�5�Yc�A�4*

loss5�<��b�       �	�>6�Yc�A�4*

loss�W=�l�       �	��6�Yc�A�4*

lossA��<�Ly$       �	p7�Yc�A�4*

loss��r='U��       �	�8�Yc�A�4*

loss6��<TAo�       �	'�8�Yc�A�4*

loss*cy=)ܱ       �	`;9�Yc�A�4*

loss3}�<_1�	       �	��9�Yc�A�4*

loss���<ߟ�4       �	fk:�Yc�A�4*

loss�`�<P-�j       �	%;�Yc�A�4*

loss1*�;s>d�       �	H�;�Yc�A�4*

loss�ɀ<[)aX       �	XY<�Yc�A�4*

loss�=�Q,Z       �	��<�Yc�A�4*

lossM�<��+�       �	��=�Yc�A�4*

loss�M�<�ͻ�       �	�0>�Yc�A�4*

loss�]%=2�       �	��>�Yc�A�4*

loss���<C2y       �	̘?�Yc�A�4*

loss���; V�Z       �	>B@�Yc�A�4*

loss�<ӡD�       �	��@�Yc�A�4*

loss#��<�x��       �	�}A�Yc�A�4*

loss��=�/       �	iTB�Yc�A�4*

loss�P:=���v       �	�B�Yc�A�4*

loss��=�(p       �	�C�Yc�A�4*

loss�~�<�,y       �	�#D�Yc�A�4*

lossQR=C� �       �	]�D�Yc�A�4*

loss��;
A<>       �	�[E�Yc�A�4*

loss���<��G       �	s�E�Yc�A�4*

loss��!=ZlA�       �	a�F�Yc�A�4*

lossA/�=�?�
       �	�%G�Yc�A�4*

loss��d<Q$��       �	��G�Yc�A�4*

loss�۷<δ�M       �	tH�Yc�A�4*

loss1�"<� b[       �	FI�Yc�A�4*

loss_p�<.E�       �	0�I�Yc�A�4*

loss��:=Qb       �	[xJ�Yc�A�4*

lossd�"=�B��       �	K�Yc�A�4*

loss�&<a&x       �	��K�Yc�A�4*

loss��;��       �	7PL�Yc�A�4*

loss'K=�;_       �	��L�Yc�A�4*

loss8{<q���       �	��M�Yc�A�4*

lossn!�<ԛ�&       �	�@N�Yc�A�4*

loss���<�̝�       �	��N�Yc�A�4*

loss{�<��z`       �	�tO�Yc�A�4*

loss_=���W       �	TP�Yc�A�4*

losst"�<��       �	c�P�Yc�A�4*

loss�9�<4���       �	uZQ�Yc�A�4*

losslW<���       �	��Q�Yc�A�4*

loss?s)=���       �	Z�R�Yc�A�4*

loss���<��=       �	�S�Yc�A�4*

loss�u�<���       �	�!T�Yc�A�4*

loss��<����       �	;�T�Yc�A�4*

lossS�;�T�       �	W`U�Yc�A�4*

loss�u;=�
��       �	��U�Yc�A�4*

loss\e�<�Ϫ       �	עV�Yc�A�4*

loss?H�;@U��       �	 :W�Yc�A�5*

loss/�.<��z       �	��W�Yc�A�5*

loss���;#�z       �	w�X�Yc�A�5*

loss��=6/�       �	�"Y�Yc�A�5*

loss�_B=v�m�       �	��Y�Yc�A�5*

loss��$=�1t�       �	�RZ�Yc�A�5*

loss��=�/�       �	V�Z�Yc�A�5*

loss�=�Fī       �	��[�Yc�A�5*

loss.��<	4��       �	5C\�Yc�A�5*

loss*Ş<]��,       �	��\�Yc�A�5*

loss�#�<�#�t       �	�}]�Yc�A�5*

loss/�4<���u       �	@m^�Yc�A�5*

loss=ȝB       �	p
_�Yc�A�5*

loss�(?<�֥l       �	��_�Yc�A�5*

loss�G=Zo�       �	�F`�Yc�A�5*

loss?A�=v	\�       �	��`�Yc�A�5*

loss1�;/���       �	�wa�Yc�A�5*

lossL�0=K�
       �	3b�Yc�A�5*

loss�{<N�sT       �	f�b�Yc�A�5*

loss�c�<�W�!       �	�Xc�Yc�A�5*

loss3%�=mw��       �	��c�Yc�A�5*

loss�+T<���f       �	�d�Yc�A�5*

loss�$�<�gh�       �	�ae�Yc�A�5*

loss_��<�(�       �	�Gf�Yc�A�5*

loss��i<`|�(       �	��f�Yc�A�5*

loss� =h�GJ       �	 zg�Yc�A�5*

loss)�<y�/f       �	p&h�Yc�A�5*

lossm� =qOW       �	�h�Yc�A�5*

lossMI�<���       �	SZi�Yc�A�5*

lossdQ<�#kc       �	l	j�Yc�A�5*

loss䟾<���       �	~�j�Yc�A�5*

loss��i=o�5�       �	�Tk�Yc�A�5*

loss@i=Rcq       �	��k�Yc�A�5*

loss�=��~)       �	A�l�Yc�A�5*

loss��=D��F       �	�Um�Yc�A�5*

loss:�=�cۊ       �	,dn�Yc�A�5*

loss�[ =�2.�       �	o�Yc�A�5*

lossq<6`��       �	p�Yc�A�5*

losswl�<�N�       �	��p�Yc�A�5*

loss7��<����       �	�_q�Yc�A�5*

loss�e=�|       �	lr�Yc�A�5*

loss)��<���.       �	x�r�Yc�A�5*

loss4�<�_��       �	��s�Yc�A�5*

loss�4�<�Qź       �	dt�Yc�A�5*

loss� �<�=       �	��t�Yc�A�5*

lossC�A<5��       �	IJu�Yc�A�5*

loss���;�`c�       �	��u�Yc�A�5*

loss��<l2�       �	��v�Yc�A�5*

loss��<�%��       �	->w�Yc�A�5*

loss�J=��,       �	Vx�Yc�A�5*

loss��<ʟѦ       �	��x�Yc�A�5*

lossD=\��       �	�Yy�Yc�A�5*

loss:�;=/��       �	�z�Yc�A�5*

loss��=r��       �	��z�Yc�A�5*

loss�}�<����       �	kJ{�Yc�A�5*

loss7�
=Dy@�       �	F�{�Yc�A�5*

losst�=<�܅       �	 �|�Yc�A�5*

loss���<j12�       �	|)}�Yc�A�5*

lossO�'=5x��       �	)�}�Yc�A�5*

loss��<�       �	l~�Yc�A�5*

loss���<�͌       �	��Yc�A�5*

loss�y�<9̴4       �	N��Yc�A�5*

loss��<x�AY       �	�a��Yc�A�5*

loss�<),_       �	p��Yc�A�5*

lossC�<5��       �	����Yc�A�5*

loss�B�<�.�       �	�f��Yc�A�5*

loss/=�<���       �	�
��Yc�A�5*

loss�t=�W��       �	����Yc�A�5*

loss��V=�@��       �	؀��Yc�A�5*

loss���<-�k�       �	9'��Yc�A�5*

loss�<��I)       �	 ą�Yc�A�5*

loss)�'<m.S       �	l_��Yc�A�5*

lossԯ�;3W��       �	R��Yc�A�5*

loss���<7;4l       �	����Yc�A�5*

loss��= %�V       �	�_��Yc�A�5*

loss,"<썽
       �	q��Yc�A�5*

lossL>n<-ў       �	���Yc�A�5*

loss�<�r�       �	�W��Yc�A�5*

loss�L�<���       �	DL��Yc�A�5*

loss�Q?=����       �	Q���Yc�A�5*

loss̀"=Q4       �	␌�Yc�A�5*

loss_f=e��       �	�@��Yc�A�5*

loss���;�(�       �	���Yc�A�5*

loss�a�=?��	       �	Ȗ��Yc�A�5*

lossҡ<�=u�       �	�ԏ�Yc�A�5*

loss��f=Ѿ�u       �	Bw��Yc�A�5*

loss_x"=�_�[       �	y$��Yc�A�5*

loss*K�<���C       �	rő�Yc�A�5*

loss��<�z��       �	Vd��Yc�A�5*

loss��4=5
��       �	���Yc�A�5*

loss-�=mT��       �	���Yc�A�5*

loss���=4���       �	2W��Yc�A�5*

loss�ʃ=�a0a       �	���Yc�A�5*

loss?!-=
A\1       �	����Yc�A�5*

loss.7�;�o��       �	_���Yc�A�5*

loss�r<?�'�       �	�S��Yc�A�5*

lossF�=���0       �	����Yc�A�5*

loss&|<�2       �	���Yc�A�5*

lossh[7<OF       �	�^��Yc�A�5*

loss�`�<yF}�       �	���Yc�A�5*

loss3�=��1       �	�a��Yc�A�5*

loss�~[=r	�       �	���Yc�A�5*

lossT�W=�-!�       �	����Yc�A�5*

lossSA=����       �	�h��Yc�A�5*

loss��=���B       �	���Yc�A�5*

lossvU=�$YN       �	����Yc�A�5*

loss�.�;�_�9       �	�Z��Yc�A�5*

loss�<yv#w       �	]���Yc�A�5*

lossɧ�<�f:�       �	����Yc�A�5*

loss:�.<lϻ�       �	QM��Yc�A�5*

loss��%;���       �	���Yc�A�5*

loss��<o��	       �	����Yc�A�5*

loss�c=CC�{       �	pD��Yc�A�5*

losse��<��4       �	���Yc�A�5*

loss=%�k�       �	0���Yc�A�5*

loss�"j=�ƭ�       �	]���Yc�A�5*

lossc�=�i=�       �	�*��Yc�A�5*

loss{"�<}5��       �	�d��Yc�A�5*

loss�#=��`�       �	���Yc�A�5*

loss���<�1f       �	���Yc�A�5*

lossM=�c��       �	�F��Yc�A�5*

loss
��<�P�       �	���Yc�A�5*

losse�=:]N       �	7��Yc�A�5*

loss��;)��       �	���Yc�A�5*

loss5�!=�2C8       �	��Yc�A�5*

loss��{<�fq�       �	77��Yc�A�5*

loss��<W���       �	���Yc�A�5*

loss�jV<')�^       �	9~��Yc�A�5*

loss1�=���s       �	�\��Yc�A�6*

loss_d�<3��       �	����Yc�A�6*

loss��<֯��       �	:���Yc�A�6*

lossD,)<B�w�       �	����Yc�A�6*

loss��=+�.K       �	�O��Yc�A�6*

loss�\�<�<>       �	I��Yc�A�6*

loss׻j<�1�       �	�´�Yc�A�6*

loss.�8=.{       �	�]��Yc�A�6*

loss_P�<�椌       �	����Yc�A�6*

loss[-�<E�/0       �	׿��Yc�A�6*

loss�H=o�b       �	�[��Yc�A�6*

loss�>D=�H�       �	j���Yc�A�6*

loss�F%<��P�       �	�*��Yc�A�6*

loss�O=�;       �	�Ĺ�Yc�A�6*

lossv1�<3Y�       �	Z~��Yc�A�6*

lossƵ�<�W�       �	���Yc�A�6*

loss�c=]O       �	�"��Yc�A�6*

loss.��<��       �	����Yc�A�6*

lossL�)=� ��       �	<f��Yc�A�6*

loss\KV<c���       �	u��Yc�A�6*

loss��<d�q       �	���Yc�A�6*

loss���;���.       �	 7��Yc�A�6*

loss��P<��3       �	�ӿ�Yc�A�6*

loss�}n='��       �	�o��Yc�A�6*

loss{
=��#�       �	o��Yc�A�6*

loss a=�<�       �	����Yc�A�6*

loss�ov<�o:�       �	p>��Yc�A�6*

loss/5=��Ү       �	Z���Yc�A�6*

lossf�<�ƌ       �	y���Yc�A�6*

loss��;;��l(       �	�4��Yc�A�6*

lossR<��D,       �	8���Yc�A�6*

loss��<�Gh�       �	����Yc�A�6*

loss��<�/�Q       �	*��Yc�A�6*

lossN=��       �	����Yc�A�6*

loss��?=��hm       �	�j��Yc�A�6*

loss��<�PYG       �	���Yc�A�6*

loss�,=�a�%       �	����Yc�A�6*

lossI�<̬       �	�;��Yc�A�6*

loss�ą=�n�       �	����Yc�A�6*

lossTwj<=�S�       �	|���Yc�A�6*

loss���<6k'       �	C:��Yc�A�6*

loss\�,<*f+�       �	����Yc�A�6*

loss}
�<�*       �	Ɗ��Yc�A�6*

loss�<� �       �	zU��Yc�A�6*

loss
��;R7n-       �	��Yc�A�6*

loss�Y<��L�       �	o���Yc�A�6*

lossr��<�:�       �	�S��Yc�A�6*

loss6�<4x�       �	����Yc�A�6*

loss�y<��       �	�;��Yc�A�6*

loss�>'=w�ˍ       �	����Yc�A�6*

loss�ط<_~       �	z���Yc�A�6*

loss8�T=����       �	�d��Yc�A�6*

loss��^="�%W       �	���Yc�A�6*

loss�eL<��8,       �	����Yc�A�6*

loss��=U��       �	]3��Yc�A�6*

loss髽<�F��       �	���Yc�A�6*

loss�H�<��d
       �	�p��Yc�A�6*

loss�R�;��J/       �	���Yc�A�6*

loss�I<�´�       �	Ϣ��Yc�A�6*

loss���<w!�       �	[A��Yc�A�6*

loss6O�;���       �	�&��Yc�A�6*

loss��&=k7�D       �	����Yc�A�6*

loss��<ف��       �	���Yc�A�6*

loss�c�<$�֜       �	2"��Yc�A�6*

loss��<a>B       �	���Yc�A�6*

loss���<�Eɫ       �	����Yc�A�6*

loss���<ƻ��       �	Jb��Yc�A�6*

loss�-=z�A       �	~���Yc�A�6*

loss6�I=6n2D       �	Û��Yc�A�6*

loss���<��V       �	81��Yc�A�6*

loss)X=F���       �	���Yc�A�6*

loss:b�;a�6       �	�f��Yc�A�6*

loss� �<Ί�       �	��Yc�A�6*

loss�7<|�,�       �	����Yc�A�6*

loss�q;,���       �	�K��Yc�A�6*

loss���<X*\�       �	l���Yc�A�6*

lossZǕ=�5�+       �	Փ��Yc�A�6*

loss�q;Ǫ       �	�:��Yc�A�6*

loss�B<d`e�       �	#1��Yc�A�6*

loss�D;<����       �	-���Yc�A�6*

loss(f�<����       �	e��Yc�A�6*

losso�\<�x#       �	? ��Yc�A�6*

loss�9=~Bw�       �	/���Yc�A�6*

loss[�n<p�       �	g��Yc�A�6*

loss��%=�d       �	���Yc�A�6*

loss��<�<�       �	r���Yc�A�6*

lossW�e<\z׫       �	6?��Yc�A�6*

loss��<����       �	����Yc�A�6*

lossH�$=��MW       �	����Yc�A�6*

lossmd�<�
>       �	��Yc�A�6*

loss��I=���z       �	e���Yc�A�6*

loss���<�;�       �	k���Yc�A�6*

loss8��<z�&2       �	T��Yc�A�6*

loss{�<*y�       �	D���Yc�A�6*

loss�P�<�Ν�       �	����Yc�A�6*

loss�B�<��       �	�;��Yc�A�6*

lossm�[=Ǔ�&       �	{���Yc�A�6*

loss���;9��o       �	�q��Yc�A�6*

loss�J=�,       �	b��Yc�A�6*

loss��6=���}       �	2���Yc�A�6*

loss�`
=�q6d       �	JB��Yc�A�6*

loss@0<����       �	+���Yc�A�6*

loss�aq=�a"o       �	����Yc�A�6*

lossj�{=΢Z       �	�&��Yc�A�6*

loss%l=���       �	����Yc�A�6*

loss|�<dk{�       �	�r��Yc�A�6*

loss�[2<�       �	��Yc�A�6*

loss�Ƭ<-�       �	���Yc�A�6*

losskp=�f�       �	kf��Yc�A�6*

loss��d<���K       �	& ��Yc�A�6*

loss�q<��5�       �	����Yc�A�6*

loss�;�%S,       �	,E��Yc�A�6*

loss�Xr=i�-�       �	����Yc�A�6*

loss=E5=��,�       �	@���Yc�A�6*

lossw=j�ߗ       �	�.��Yc�A�6*

loss�O�<�Ϟa       �	}���Yc�A�6*

loss8�=�h�       �	Ln��Yc�A�6*

loss�=�U�       �	;��Yc�A�6*

loss�W<cV�U       �	����Yc�A�6*

loss��<e.Φ       �	io��Yc�A�6*

loss���<?��       �	b �Yc�A�6*

loss��P<Wl�       �	� �Yc�A�6*

loss��<��2i       �	�^�Yc�A�6*

loss��<�x��       �	G�Yc�A�6*

loss���<�I�R       �	D��Yc�A�6*

loss�$�<,�T5       �	�z�Yc�A�6*

loss�}i<��       �	��Yc�A�6*

loss<�<!�$�       �	b��Yc�A�6*

loss���;��*�       �	�W�Yc�A�7*

loss��T=�f��       �	��Yc�A�7*

loss��=���       �	!��Yc�A�7*

loss2q<� �       �	='�Yc�A�7*

loss3ŷ<��       �	���Yc�A�7*

lossC�><TqVF       �	d�Yc�A�7*

loss���<���       �	�	�Yc�A�7*

loss��=�Cҳ       �	�	�Yc�A�7*

loss�U�<�b       �	a
�Yc�A�7*

lossVu:=B�}c       �	��
�Yc�A�7*

loss\ׇ<�x�       �	J��Yc�A�7*

lossj5=�E'm       �	8�Yc�A�7*

lossQ�)<��K]       �	���Yc�A�7*

loss�9<�Ì�       �	Ql�Yc�A�7*

loss�}�<�کk       �	�Yc�A�7*

loss�XG=���x       �	���Yc�A�7*

lossTH"=�ZJ       �	K#�Yc�A�7*

loss,c�<p�+       �	
�Yc�A�7*

loss͐�<�'8       �	���Yc�A�7*

loss��9<��P       �	�n�Yc�A�7*

lossj4�<
ߧ�       �	:�Yc�A�7*

loss�Q <kL(�       �	E�Yc�A�7*

loss�g�;7 @8       �	�{�Yc�A�7*

loss���;%���       �	�X�Yc�A�7*

loss�`7=��\       �	�G�Yc�A�7*

lossgF�<�H��       �	�1�Yc�A�7*

loss�} =��\       �	�Yc�A�7*

loss_0=�{�       �	���Yc�A�7*

loss.�<�K��       �	�Q�Yc�A�7*

loss��Y<|�:�       �	���Yc�A�7*

lossʝ�;�z%�       �	��Yc�A�7*

lossn��<���       �	���Yc�A�7*

lossJS�;Q$��       �	2��Yc�A�7*

lossq��<F�a�       �	�.�Yc�A�7*

loss���< p�I       �	g� �Yc�A�7*

lossF�<^�-~       �	�h!�Yc�A�7*

loss��;��?�       �	�"�Yc�A�7*

lossV7J;��&       �	i�"�Yc�A�7*

lossZ��<�6�       �	�E#�Yc�A�7*

loss�&D<t�       �	�#�Yc�A�7*

loss_��=\yJ�       �	|$�Yc�A�7*

loss�cz=1�UP       �	�%�Yc�A�7*

lossD"�<���`       �	̸%�Yc�A�7*

loss�:�<r�sZ       �	�k&�Yc�A�7*

loss�y�<e.�       �	A'�Yc�A�7*

lossש�<��V       �	j�'�Yc�A�7*

loss~�<2W&�       �	�x(�Yc�A�7*

loss���<�F~       �	�)�Yc�A�7*

loss��<�G*       �	�)�Yc�A�7*

lossF�)=X�d�       �	�Q*�Yc�A�7*

lossqL	=�v��       �	��*�Yc�A�7*

lossءv<B�U�       �	��+�Yc�A�7*

lossh��<�_�X       �	�5,�Yc�A�7*

loss���;w�       �	��,�Yc�A�7*

lossn�<K��6       �	�r-�Yc�A�7*

loss�3I<��<{       �	c.�Yc�A�7*

loss�u�;%9��       �	/�.�Yc�A�7*

loss_:j<���       �	:?/�Yc�A�7*

loss�>�<!��       �	��/�Yc�A�7*

loss�Y<��2�       �	�n0�Yc�A�7*

loss�e0=N{�       �	�1�Yc�A�7*

loss�=���       �	¡1�Yc�A�7*

loss�m#<�e�{       �	.92�Yc�A�7*

lossŕ?;��       �	��2�Yc�A�7*

loss��;G       �	�h3�Yc�A�7*

lossm��<'�H       �	� 4�Yc�A�7*

loss�8<C��k       �	I�4�Yc�A�7*

loss�j<X;�       �	F5�Yc�A�7*

loss3?<s�m�       �	��5�Yc�A�7*

loss�ih=$��)       �	Sz6�Yc�A�7*

lossIh�<}�       �	<7�Yc�A�7*

lossO�<3��       �	��7�Yc�A�7*

losss2='�$�       �	�I8�Yc�A�7*

loss�Ԫ;Ej�       �	2�8�Yc�A�7*

loss�!�<�Ȯ        �	>y9�Yc�A�7*

lossO�<�}z3       �	:�Yc�A�7*

lossz�k<ǅ�       �	/�:�Yc�A�7*

loss,��<'�%n       �	F@;�Yc�A�7*

loss�<<*@T�       �	E<�Yc�A�7*

loss��;��J       �	��<�Yc�A�7*

loss�/ =�5S]       �	�R=�Yc�A�7*

loss�,"<A��M       �	��=�Yc�A�7*

loss6�<I>E�       �	G�>�Yc�A�7*

loss=��;�<M       �	�'?�Yc�A�7*

loss��~=Գ        �	+�?�Yc�A�7*

loss�>+=]Dii       �	�X@�Yc�A�7*

loss��,<!�K'       �	�A�Yc�A�7*

loss:Zw=G��&       �	��A�Yc�A�7*

loss�Q"=m�P       �	,EB�Yc�A�7*

loss.�<u7J�       �	�C�Yc�A�7*

lossm�;��`       �	��D�Yc�A�7*

loss���;�p�       �	�-E�Yc�A�7*

loss��)=�WW       �	f�E�Yc�A�7*

loss}��;�sՂ       �	z�F�Yc�A�7*

loss:J�:�N"2       �	|DG�Yc�A�7*

lossV��<��G       �	!�G�Yc�A�7*

loss�ER;��t
       �	��H�Yc�A�7*

loss_Dz;�ʏ3       �	j2I�Yc�A�7*

loss%(�;k@�c       �	o�I�Yc�A�7*

lossn_F<o���       �	�oJ�Yc�A�7*

lossJV<G7�       �	8K�Yc�A�7*

loss��<�8w�       �	��K�Yc�A�7*

loss6=�9�L1u       �	�OL�Yc�A�7*

lossi3c:R�C       �	�L�Yc�A�7*

loss/u= 9�       �	�}M�Yc�A�7*

lossE�<'yR�       �	�PN�Yc�A�7*

loss;8<�"��       �	�KO�Yc�A�7*

loss��<��       �	~P�Yc�A�7*

losst��<�T,�       �	�4Q�Yc�A�7*

loss���=��       �	�R�Yc�A�7*

loss��/<�9t       �	��R�Yc�A�7*

loss���;U���       �	��S�Yc�A�7*

lossa4=���       �	�mT�Yc�A�7*

loss��w<�Ml�       �	�U�Yc�A�7*

loss£<���       �	��U�Yc�A�7*

lossӋj=(�       �	m�V�Yc�A�7*

loss{��<f��V       �	�*W�Yc�A�7*

loss❅=�%?       �	h�W�Yc�A�7*

loss!;�<%V�       �	�gX�Yc�A�7*

loss[�=Rz       �	�Y�Yc�A�7*

loss��)=w���       �	��Y�Yc�A�7*

loss*"�=Y�<�       �	�|Z�Yc�A�7*

loss/�]=q((       �	�[�Yc�A�7*

lossp�<����       �	�[�Yc�A�7*

loss�u�<���       �	HP\�Yc�A�7*

loss�v�<}�u       �	��\�Yc�A�7*

lossr�O<:�f�       �	]�]�Yc�A�7*

lossTu+=����       �	2^�Yc�A�7*

loss,H�=t���       �	^�^�Yc�A�8*

loss��7<�.��       �	�k_�Yc�A�8*

loss��<<GG��       �	�`�Yc�A�8*

loss��<}Tȁ       �	d\a�Yc�A�8*

lossT��;Xh�       �	
�a�Yc�A�8*

lossc`�;n�k�       �	M�b�Yc�A�8*

loss��=�P�-       �	�Bc�Yc�A�8*

lossϚL;O��       �	�c�Yc�A�8*

loss��<���       �	��d�Yc�A�8*

loss��<b"/E       �	>!e�Yc�A�8*

loss�*=� �       �	��e�Yc�A�8*

loss�w=����       �	�mf�Yc�A�8*

loss:b<jb_O       �	�g�Yc�A�8*

lossR�Y<"�8       �	��g�Yc�A�8*

loss1��;�7�       �	�jh�Yc�A�8*

loss)�X;��ؼ       �	�%i�Yc�A�8*

loss��<U���       �	��i�Yc�A�8*

loss���<o|�       �	vj�Yc�A�8*

loss�F<���       �	�k�Yc�A�8*

loss���<�2��       �	Z�k�Yc�A�8*

lossV&"=��t       �	Xl�Yc�A�8*

loss�=���       �	�m�Yc�A�8*

loss�Ê;��H�       �	�m�Yc�A�8*

loss�d<#@$       �	�n�Yc�A�8*

lossZ�%<U�       �	4-o�Yc�A�8*

lossv�#=+Z        �	��o�Yc�A�8*

loss��Z<�&�       �	�p�Yc�A�8*

loss2! =ܿ)^       �	Lq�Yc�A�8*

loss	=Zۥ�       �	o�q�Yc�A�8*

lossۤI;q��o       �	��r�Yc�A�8*

lossΌ�<)?�       �	�Fs�Yc�A�8*

loss,��;���       �	)�s�Yc�A�8*

loss,?
<-$�^       �	��t�Yc�A�8*

loss�K�<Ih�