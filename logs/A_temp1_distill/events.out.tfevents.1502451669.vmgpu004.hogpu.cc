       �K"	  @�ec�Abrain.Event:2����"�     �])�	6�o�ec�A"��
^
dataPlaceholder*/
_output_shapes
:���������*
shape: *
dtype0
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
shape: *
dtype0
v
conv2d_1/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
`
conv2d_1/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�x�
`
conv2d_1/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�x=
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
dtype0*
seed���)*
T0*&
_output_shapes
:@*
seed2��l
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
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
:@
�
conv2d_1/kernel
VariableV2*&
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
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
conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0
[
conv2d_1/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
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
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_1/bias/readIdentityconv2d_1/bias*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0
s
conv2d_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
s
"conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
use_cudnn_on_gpu(*/
_output_shapes
:���������@*
data_formatNHWC*
strides
*
T0*
paddingVALID
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
conv2d_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
`
conv2d_2/random_uniform/minConst*
valueB
 *�\1�*
_output_shapes
: *
dtype0
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
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 
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
VariableV2*
shape:@@*
shared_name *
dtype0*&
_output_shapes
:@@*
	container 
�
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
[
conv2d_2/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
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
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:���������@*
data_formatNHWC*
strides

�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:���������@
e
activation_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
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
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0
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
 *  �?*
_output_shapes
: *
dtype0
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2�ޫ*
T0*
seed���)*
dtype0
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
dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
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
N*
T0
c
flatten_1/ShapeShapedropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask *
T0*
Index0*
end_mask*
_output_shapes
:*

begin_mask *
ellipsis_mask 
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
���������*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*

axis *
_output_shapes
:*
T0*
N
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
Tshape0*0
_output_shapes
:������������������*
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
seed2ȁ�*
T0*
seed���)*
dtype0
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
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*!
_output_shapes
:���
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
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
data_formatNHWC*
T0
]
activation_3/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:����������
�
dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
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
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
dtype0*
seed���)*
T0*(
_output_shapes
:����������*
seed2���
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
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
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
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
seed���)*
T0*
dtype0*
_output_shapes
:	�
*
seed2��,
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
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	�
*
T0
�
dense_2/kernel
VariableV2*
_output_shapes
:	�
*
	container *
shape:	�
*
dtype0*
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�

|
dense_2/kernel/readIdentitydense_2/kernel*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
Z
dense_2/ConstConst*
valueB
*    *
_output_shapes
:
*
dtype0
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
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

q
dense_2/bias/readIdentitydense_2/bias*
_output_shapes
:
*
_class
loc:@dense_2/bias*
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
'sequential_1/conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
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
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*
T0*/
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
dtype0*
_output_shapes
:*
valueB"      
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*
use_cudnn_on_gpu(*/
_output_shapes
:���������@*
strides
*
data_formatNHWC*
T0*
paddingVALID
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@

sequential_1/activation_2/ReluRelusequential_1/conv2d_2/BiasAdd*
T0*/
_output_shapes
:���������@
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
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
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
seed2���
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
)sequential_1/dropout_1/cond/dropout/FloorFloor'sequential_1/dropout_1/cond/dropout/add*/
_output_shapes
:���������@*
T0
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
T0*1
_class'
%#loc:@sequential_1/activation_2/Relu*J
_output_shapes8
6:���������@:���������@
�
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
T0*
N
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
t
*sequential_1/flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
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
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
_output_shapes
:*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask *
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
"sequential_1/dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

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
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2ƶ�*
dtype0*
T0*
seed���)
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
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
)sequential_1/dropout_2/cond/dropout/FloorFloor'sequential_1/dropout_2/cond/dropout/add*(
_output_shapes
:����������*
T0
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
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul**
_output_shapes
:����������: *
N*
T0
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
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
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
num_inst/readIdentitynum_inst*
T0*
_output_shapes
: *
_class
loc:@num_inst
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
num_correct/readIdentitynum_correct*
_output_shapes
: *
_class
loc:@num_correct*
T0
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
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
X
SumSumToFloatConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �B
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
Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *    
�
Assign_1Assignnum_correctConst_3*
_output_shapes
: *
validate_shape(*
_class
loc:@num_correct*
T0*
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
 *  �?
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*
T0*'
_output_shapes
:���������

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
T0*
out_type0*
_output_shapes
:
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
out_type0*
_output_shapes
:*
T0
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
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*

axis *
_output_shapes
:*
T0*
N
o
%softmax_cross_entropy_loss/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
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
&softmax_cross_entropy_loss/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
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
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
N*
T0*
_output_shapes
:*

axis 
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
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
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
_output_shapes
: *
T0
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
�
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*
T0*
_output_shapes
:*

axis 
�
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*#
_output_shapes
:���������*
Index0*
T0
�
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*#
_output_shapes
:���������*
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
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
value	B :*
_output_shapes
: *
dtype0
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
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
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
T0*
_output_shapes
: 
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
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
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
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
_output_shapes
: *
dtype0
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
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
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
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
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
]
PlaceholderPlaceholder*'
_output_shapes
:���������
*
shape: *
dtype0
L
div_2/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
i
div_2RealDivsequential_1/dense_2/BiasAdddiv_2/y*
T0*'
_output_shapes
:���������

c
!softmax_cross_entropy_loss_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
g
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
_output_shapes
:*
out_type0*
T0
e
#softmax_cross_entropy_loss_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
out_type0*
_output_shapes
:*
T0
d
"softmax_cross_entropy_loss_1/Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss_1/SubSub#softmax_cross_entropy_loss_1/Rank_1"softmax_cross_entropy_loss_1/Sub/y*
_output_shapes
: *
T0
�
(softmax_cross_entropy_loss_1/Slice/beginPack softmax_cross_entropy_loss_1/Sub*
N*
T0*
_output_shapes
:*

axis 
q
'softmax_cross_entropy_loss_1/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
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
(softmax_cross_entropy_loss_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
�
$softmax_cross_entropy_loss_1/ReshapeReshapediv_2#softmax_cross_entropy_loss_1/concat*
T0*0
_output_shapes
:������������������*
Tshape0
e
#softmax_cross_entropy_loss_1/Rank_2Const*
value	B :*
_output_shapes
: *
dtype0
o
$softmax_cross_entropy_loss_1/Shape_2ShapePlaceholder*
T0*
_output_shapes
:*
out_type0
f
$softmax_cross_entropy_loss_1/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
�
"softmax_cross_entropy_loss_1/Sub_1Sub#softmax_cross_entropy_loss_1/Rank_2$softmax_cross_entropy_loss_1/Sub_1/y*
T0*
_output_shapes
: 
�
*softmax_cross_entropy_loss_1/Slice_1/beginPack"softmax_cross_entropy_loss_1/Sub_1*
_output_shapes
:*
N*

axis *
T0
s
)softmax_cross_entropy_loss_1/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
$softmax_cross_entropy_loss_1/Slice_1Slice$softmax_cross_entropy_loss_1/Shape_2*softmax_cross_entropy_loss_1/Slice_1/begin)softmax_cross_entropy_loss_1/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
.softmax_cross_entropy_loss_1/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
l
*softmax_cross_entropy_loss_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
%softmax_cross_entropy_loss_1/concat_1ConcatV2.softmax_cross_entropy_loss_1/concat_1/values_0$softmax_cross_entropy_loss_1/Slice_1*softmax_cross_entropy_loss_1/concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
value	B :*
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss_1/Sub_2Sub!softmax_cross_entropy_loss_1/Rank$softmax_cross_entropy_loss_1/Sub_2/y*
T0*
_output_shapes
: 
t
*softmax_cross_entropy_loss_1/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
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
&softmax_cross_entropy_loss_1/Reshape_2Reshape%softmax_cross_entropy_loss_1/xentropy$softmax_cross_entropy_loss_1/Slice_2*#
_output_shapes
:���������*
Tshape0*
T0
~
9softmax_cross_entropy_loss_1/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
?softmax_cross_entropy_loss_1/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
>softmax_cross_entropy_loss_1/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
U
Msoftmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successNoOp
�
(softmax_cross_entropy_loss_1/ToFloat_1/xConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
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
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
.softmax_cross_entropy_loss_1/num_present/EqualEqual(softmax_cross_entropy_loss_1/ToFloat_1/x0softmax_cross_entropy_loss_1/num_present/Equal/y*
_output_shapes
: *
T0
�
3softmax_cross_entropy_loss_1/num_present/zeros_like	ZerosLike(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*
_output_shapes
: 
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
2softmax_cross_entropy_loss_1/num_present/ones_likeFill8softmax_cross_entropy_loss_1/num_present/ones_like/Shape8softmax_cross_entropy_loss_1/num_present/ones_like/Const*
T0*
_output_shapes
: 
�
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
_output_shapes
: *
T0
�
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
�
ksoftmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
valueB: *
_output_shapes
:*
dtype0
�
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
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
"softmax_cross_entropy_loss_1/EqualEqual(softmax_cross_entropy_loss_1/num_present$softmax_cross_entropy_loss_1/Equal/y*
T0*
_output_shapes
: 
�
,softmax_cross_entropy_loss_1/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
,softmax_cross_entropy_loss_1/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
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
 softmax_cross_entropy_loss_1/divRealDiv"softmax_cross_entropy_loss_1/Sum_1#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
y
'softmax_cross_entropy_loss_1/zeros_like	ZerosLike"softmax_cross_entropy_loss_1/Sum_1*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/valueSelect$softmax_cross_entropy_loss_1/Greater softmax_cross_entropy_loss_1/div'softmax_cross_entropy_loss_1/zeros_like*
T0*
_output_shapes
: 
P
Placeholder_1Placeholder*
dtype0*
shape: *
_output_shapes
:
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
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select
�
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1*
_output_shapes
: 
x
5gradients/softmax_cross_entropy_loss_1/div_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
z
7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
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
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
@gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape*
_output_shapes
: *
T0
�
Jgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1*
_output_shapes
: *
T0
�
=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like	ZerosLike&softmax_cross_entropy_loss_1/ones_like*
T0*
_output_shapes
: 
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
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
_output_shapes
: *N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
T0
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
T0*
_output_shapes
: *
Tshape0
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
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
7gradients/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
�
5gradients/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
_output_shapes
:*
out_type0*
T0
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
T0*
Tshape0*
_output_shapes
:
�
=gradients/softmax_cross_entropy_loss_1/num_present_grad/ShapeShape:softmax_cross_entropy_loss_1/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
�
<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileTile?gradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape=gradients/softmax_cross_entropy_loss_1/num_present_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
out_type0*
_output_shapes
:
z
7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Egradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulMul4gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*#
_output_shapes
:���������
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
5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_1Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Ggradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_17gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
valueB *
dtype0*
_output_shapes
: 
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
_output_shapes
:*
out_type0*
T0
�
_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mulMul<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumSumMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1Mul/softmax_cross_entropy_loss_1/num_present/Select<gradients/softmax_cross_entropy_loss_1/num_present_grad/Tile*
T0*#
_output_shapes
:���������
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*#
_output_shapes
:���������*
Tshape0
�
Zgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_depsNoOpR^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeT^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
bgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
�
dgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentitySgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������*
T0
�
Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
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
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
�
gradients/zeros_like	ZerosLike'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
�
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
�
Cgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
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
T0*
_output_shapes
:*
out_type0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
gradients/div_2_grad/RealDivRealDiv;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapediv_2/y*
T0*'
_output_shapes
:���������

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
gradients/div_2_grad/Reshape_1Reshapegradients/div_2_grad/Sum_1gradients/div_2_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
/gradients/div_2_grad/tuple/control_dependency_1Identitygradients/div_2_grad/Reshape_1&^gradients/div_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/div_2_grad/Reshape_1*
_output_shapes
: *
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_2_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:

�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_2_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_2_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
_output_shapes
:���������
*
T0
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
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	�
*
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
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_1Shapegradients/Switch:1*
_output_shapes
:*
out_type0*
T0
Z
gradients/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_2/cond/dropout/divKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
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
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape*(
_output_shapes
:����������*
T0
�
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
_output_shapes
: *
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
T0*
_output_shapes
:*
out_type0
y
6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Dgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_2/cond/mul_grad/Shape6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
gradients/Shape_2Shapegradients/Switch_1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*(
_output_shapes
:����������*
T0
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
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
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
T0*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape
�
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@*
T0
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*/
_output_shapes
:���������@
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
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
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
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape*/
_output_shapes
:���������@*
T0
�
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1*/
_output_shapes
:���������@*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
_output_shapes
:*
out_type0*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
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
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/NegNegsequential_1/dropout_1/cond/mul*/
_output_shapes
:���������@*
T0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Neg-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
_output_shapes
:*
out_type0*
T0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Dgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_1/cond/mul_grad/Shape6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
?gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_1/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape*/
_output_shapes
:���������@*
T0
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
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_3/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
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
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
T0
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
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
T0*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"      @   @   
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*&
_output_shapes
:@@*
data_formatNHWC*
strides
*
T0*
paddingVALID
�
Agradients/sequential_1/conv2d_2/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput
�
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
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
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*%
valueB"         @   *
_output_shapes
:*
dtype0
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*&
_output_shapes
:@*
data_formatNHWC*
strides

�
Agradients/sequential_1/conv2d_1/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput
�
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*
T0*&
_output_shapes
:@*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
n
beta2_power/readIdentitybeta2_power*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
j
zerosConst*%
valueB@*    *
dtype0*&
_output_shapes
:@
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
zeros_1Const*%
valueB@*    *
dtype0*&
_output_shapes
:@
�
conv2d_1/kernel/Adam_1
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*
shared_name *&
_output_shapes
:@*
shape:@
�
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
T
zeros_2Const*
_output_shapes
:@*
dtype0*
valueB@*    
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
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
~
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
T
zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_1/bias/Adam_1
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
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
l
zeros_4Const*&
_output_shapes
:@@*
dtype0*%
valueB@@*    
�
conv2d_2/kernel/Adam
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
shape:@@*
shared_name 
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
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
l
zeros_5Const*&
_output_shapes
:@@*
dtype0*%
valueB@@*    
�
conv2d_2/kernel/Adam_1
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
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
T
zeros_6Const*
valueB@*    *
_output_shapes
:@*
dtype0
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
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
b
zeros_9Const* 
valueB���*    *
dtype0*!
_output_shapes
:���
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
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
W
zeros_10Const*
dtype0*
_output_shapes	
:�*
valueB�*    
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
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
�
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
VariableV2*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
shape:	�
*
dtype0*
shared_name *
	container 
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
valueB	�
*    *
_output_shapes
:	�
*
dtype0
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
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0
U
zeros_14Const*
valueB
*    *
dtype0*
_output_shapes
:

�
dense_2/bias/Adam
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
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:


dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w�?
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
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
use_locking( 
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
use_locking( 
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
use_locking( *
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
use_locking( *
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
: "�����,     nKqz	�rr�ec�AJ��
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
dtype0*
shape: */
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
valueB"         @   *
_output_shapes
:*
dtype0
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
dtype0*
seed���)*
T0*&
_output_shapes
:@*
seed2��l
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
	container *
dtype0*
shared_name *
shape:@
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
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
s
conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
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
conv2d_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�\1�
`
conv2d_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�\1=
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@@*
seed2���
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 
�
conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*&
_output_shapes
:@@*
T0
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
	container *
dtype0*
shared_name *
shape:@@
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
conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
[
conv2d_2/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
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
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������@
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
 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  @?
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
out_type0*
T0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2�ޫ
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
N*
T0*1
_output_shapes
:���������@: 
c
flatten_1/ShapeShapedropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
i
flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
end_mask*

begin_mask *
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
T0*
Index0
Y
flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
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
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
_output_shapes
:*
N*

axis *
T0
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
Tshape0*0
_output_shapes
:������������������*
T0
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
dense_1/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�3z<
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed���)*
T0*!
_output_shapes
:���*
seed2ȁ�
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
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*!
_output_shapes
:���*
T0
�
dense_1/kernel
VariableV2*!
_output_shapes
:���*
	container *
shape:���*
dtype0*
shared_name 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*!
_output_shapes
:���
~
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0
\
dense_1/ConstConst*
_output_shapes	
:�*
dtype0*
valueB�*    
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
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
:����������*
data_formatNHWC*
T0
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
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*
T0*$
_class
loc:@activation_3/Relu*<
_output_shapes*
(:����������:����������

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
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
seed2���*
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
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
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
dense_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
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
dense_2/random_uniform/maxConst*
valueB
 *̈́U>*
_output_shapes
: *
dtype0
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
dtype0*
seed���)*
T0*
_output_shapes
:	�
*
seed2��,
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
VariableV2*
_output_shapes
:	�
*
	container *
dtype0*
shared_name *
shape:	�

�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�

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
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

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
*
T0*
data_formatNHWC
�
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
�
'sequential_1/conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
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
:���������@*
T0*
data_formatNHWC

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
�
'sequential_1/conv2d_2/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
�
/sequential_1/conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*
use_cudnn_on_gpu(*/
_output_shapes
:���������@*
strides
*
data_formatNHWC*
T0*
paddingVALID
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC
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
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
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
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
_output_shapes
:*
out_type0*
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
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
:���������@*
seed2���*
dtype0*
T0*
seed���)
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
T0*
N*1
_output_shapes
:���������@: 
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
_output_shapes
:*
out_type0*
T0
t
*sequential_1/flatten_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
v
,sequential_1/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
end_mask
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
sequential_1/flatten_1/ReshapeReshape!sequential_1/dropout_1/cond/Mergesequential_1/flatten_1/stack*
T0*0
_output_shapes
:������������������*
Tshape0
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
sequential_1/activation_3/ReluRelusequential_1/dense_1/BiasAdd*(
_output_shapes
:����������*
T0
�
"sequential_1/dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

y
$sequential_1/dropout_2/cond/switch_tIdentity$sequential_1/dropout_2/cond/Switch:1*
_output_shapes
:*
T0

w
$sequential_1/dropout_2/cond/switch_fIdentity"sequential_1/dropout_2/cond/Switch*
_output_shapes
:*
T0

r
#sequential_1/dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

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
seed2ƶ�
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
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
'sequential_1/dropout_2/cond/dropout/mulMul'sequential_1/dropout_2/cond/dropout/div)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu
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
num_inst/initial_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
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
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
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
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
e
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
g
ArgMax_1ArgMaxlabelArgMax_1/dimension*

Tidx0*
T0*#
_output_shapes
:���������
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
 *  �B*
_output_shapes
: *
dtype0
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
Const_2Const*
valueB
 *    *
_output_shapes
: *
dtype0
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
divRealDivnum_correct/readadd*
_output_shapes
: *
T0
L
div_1/yConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
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
 softmax_cross_entropy_loss/ShapeShapediv_1*
T0*
_output_shapes
:*
out_type0
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
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
N*
T0*
_output_shapes
:*

axis 
o
%softmax_cross_entropy_loss/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
T0*
_output_shapes
:
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*0
_output_shapes
:������������������*
Tshape0*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
_output_shapes
:*
out_type0*
T0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
T0*

axis *
N*
_output_shapes
:
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
,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
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
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
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
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*#
_output_shapes
:���������*
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
_output_shapes
:*
out_type0
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
valueB *
_output_shapes
: *
dtype0
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
valueB *
_output_shapes
: *
dtype0
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
_output_shapes
: *
dtype0
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
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
T0*
out_type0*
_output_shapes
:
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
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
_output_shapes
:*
dtype0
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
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
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
u
%softmax_cross_entropy_loss/zeros_like	ZerosLike softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
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
 *  �?*
_output_shapes
: *
dtype0
i
div_2RealDivsequential_1/dense_2/BiasAdddiv_2/y*
T0*'
_output_shapes
:���������

c
!softmax_cross_entropy_loss_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
g
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
out_type0*
_output_shapes
:*
T0
e
#softmax_cross_entropy_loss_1/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
out_type0*
_output_shapes
:*
T0
d
"softmax_cross_entropy_loss_1/Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss_1/SubSub#softmax_cross_entropy_loss_1/Rank_1"softmax_cross_entropy_loss_1/Sub/y*
_output_shapes
: *
T0
�
(softmax_cross_entropy_loss_1/Slice/beginPack softmax_cross_entropy_loss_1/Sub*
N*
T0*
_output_shapes
:*

axis 
q
'softmax_cross_entropy_loss_1/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
"softmax_cross_entropy_loss_1/SliceSlice$softmax_cross_entropy_loss_1/Shape_1(softmax_cross_entropy_loss_1/Slice/begin'softmax_cross_entropy_loss_1/Slice/size*
Index0*
T0*
_output_shapes
:

,softmax_cross_entropy_loss_1/concat/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
j
(softmax_cross_entropy_loss_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*
N*

Tidx0*
T0*
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
$softmax_cross_entropy_loss_1/Shape_2ShapePlaceholder*
out_type0*
_output_shapes
:*
T0
f
$softmax_cross_entropy_loss_1/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
"softmax_cross_entropy_loss_1/Sub_1Sub#softmax_cross_entropy_loss_1/Rank_2$softmax_cross_entropy_loss_1/Sub_1/y*
T0*
_output_shapes
: 
�
*softmax_cross_entropy_loss_1/Slice_1/beginPack"softmax_cross_entropy_loss_1/Sub_1*
N*
T0*
_output_shapes
:*

axis 
s
)softmax_cross_entropy_loss_1/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
�
$softmax_cross_entropy_loss_1/Slice_1Slice$softmax_cross_entropy_loss_1/Shape_2*softmax_cross_entropy_loss_1/Slice_1/begin)softmax_cross_entropy_loss_1/Slice_1/size*
_output_shapes
:*
Index0*
T0
�
.softmax_cross_entropy_loss_1/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
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
N*
T0*

Tidx0
�
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
value	B :*
_output_shapes
: *
dtype0
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
)softmax_cross_entropy_loss_1/Slice_2/sizePack"softmax_cross_entropy_loss_1/Sub_2*
N*
T0*
_output_shapes
:*

axis 
�
$softmax_cross_entropy_loss_1/Slice_2Slice"softmax_cross_entropy_loss_1/Shape*softmax_cross_entropy_loss_1/Slice_2/begin)softmax_cross_entropy_loss_1/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
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
?softmax_cross_entropy_loss_1/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
>softmax_cross_entropy_loss_1/assert_broadcastable/weights/rankConst*
value	B : *
_output_shapes
: *
dtype0
�
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
value	B :*
_output_shapes
: *
dtype0
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
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
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
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
2softmax_cross_entropy_loss_1/num_present/ones_likeFill8softmax_cross_entropy_loss_1/num_present/ones_like/Shape8softmax_cross_entropy_loss_1/num_present/ones_like/Const*
T0*
_output_shapes
: 
�
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
T0*
_output_shapes
: 
�
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B : 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B :
�
ksoftmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
Dsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:���������
�
:softmax_cross_entropy_loss_1/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_1/num_present/SelectDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
.softmax_cross_entropy_loss_1/num_present/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
_output_shapes
:*
dtype0
�
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss_1/Const_1ConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
"softmax_cross_entropy_loss_1/Sum_1Sum softmax_cross_entropy_loss_1/Sum$softmax_cross_entropy_loss_1/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$softmax_cross_entropy_loss_1/GreaterGreater(softmax_cross_entropy_loss_1/num_present&softmax_cross_entropy_loss_1/Greater/y*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss_1/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss_1/EqualEqual(softmax_cross_entropy_loss_1/num_present$softmax_cross_entropy_loss_1/Equal/y*
T0*
_output_shapes
: 
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
&softmax_cross_entropy_loss_1/ones_likeFill,softmax_cross_entropy_loss_1/ones_like/Shape,softmax_cross_entropy_loss_1/ones_like/Const*
T0*
_output_shapes
: 
�
#softmax_cross_entropy_loss_1/SelectSelect"softmax_cross_entropy_loss_1/Equal&softmax_cross_entropy_loss_1/ones_like(softmax_cross_entropy_loss_1/num_present*
_output_shapes
: *
T0
�
 softmax_cross_entropy_loss_1/divRealDiv"softmax_cross_entropy_loss_1/Sum_1#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
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
Placeholder_1Placeholder*
dtype0*
shape: *
_output_shapes
:
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
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
<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like	ZerosLike softmax_cross_entropy_loss_1/div*
_output_shapes
: *
T0
�
8gradients/softmax_cross_entropy_loss_1/value_grad/SelectSelect$softmax_cross_entropy_loss_1/Greatergradients/Fill<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like*
_output_shapes
: *
T0
�
:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1Select$softmax_cross_entropy_loss_1/Greater<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_likegradients/Fill*
T0*
_output_shapes
: 
�
Bgradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_depsNoOp9^gradients/softmax_cross_entropy_loss_1/value_grad/Select;^gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
�
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select*
_output_shapes
: 
�
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1*
_output_shapes
: 
x
5gradients/softmax_cross_entropy_loss_1/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
z
7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Egradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/div_grad/Shape7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
T0*
Tshape0*
_output_shapes
: 
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
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape*
T0
�
Jgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1*
T0
�
=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like	ZerosLike&softmax_cross_entropy_loss_1/ones_like*
T0*
_output_shapes
: 
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
T0*
_output_shapes
: *N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
�
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
T0*
_output_shapes
: *
Tshape0
�
@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 
�
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
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
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
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
3gradients/softmax_cross_entropy_loss_1/Mul_grad/SumSum3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulEgradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Mul&softmax_cross_entropy_loss_1/Reshape_24gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile*#
_output_shapes
:���������*
T0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_1Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Ggradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
_output_shapes
: 
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*
out_type0*
_output_shapes
:
�
_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape*
T0*
_output_shapes
: *
Tshape0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1Mul/softmax_cross_entropy_loss_1/num_present/Select<gradients/softmax_cross_entropy_loss_1/num_present_grad/Tile*#
_output_shapes
:���������*
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*#
_output_shapes
:���������*
Tshape0
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
dgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentitySgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
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
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
out_type0*
_output_shapes
:*
T0
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
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
�
Cgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mulMul?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDimsDgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
~
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
_output_shapes
:*
out_type0*
T0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
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
gradients/div_2_grad/mulMul;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapegradients/div_2_grad/RealDiv_2*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/Sum_1Sumgradients/div_2_grad/mul,gradients/div_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_2_grad/Reshape_1Reshapegradients/div_2_grad/Sum_1gradients/div_2_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
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
T0*1
_class'
%#loc:@gradients/div_2_grad/Reshape_1*
_output_shapes
: 
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_2_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:

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
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

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
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_1Shapegradients/Switch:1*
_output_shapes
:*
out_type0*
T0
Z
gradients/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
_output_shapes
:*
out_type0*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0
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
T0*(
_output_shapes
:����������*
Tshape0
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
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape
�
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
T0*
_output_shapes
:*
out_type0
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
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
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
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
Ggradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape
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
Dgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_2/cond/mul_grad/Shape6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Mul(sequential_1/dropout_2/cond/mul/Switch:1Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
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
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
_output_shapes
: 
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
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*(
_output_shapes
:����������*
T0
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1*
T0*
N**
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
:�*
data_formatNHWC*
T0
�
<gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_3/Relu_grad/ReluGrad8^gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad
�
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
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
transpose_b( *(
_output_shapes
:����������*
transpose_a(*
T0
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
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*!
_output_shapes
:���
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
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
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
T0*
out_type0*
_output_shapes
:
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
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*/
_output_shapes
:���������@*
Tshape0*
T0
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
T0*/
_output_shapes
:���������@*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
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
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/NegNegsequential_1/dropout_1/cond/mul*/
_output_shapes
:���������@*
T0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Neg-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
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
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape*/
_output_shapes
:���������@*
T0
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
Dgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_1/cond/mul_grad/Shape6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Mul(sequential_1/dropout_1/cond/mul/Switch:1Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency*
T0*/
_output_shapes
:���������@
�
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
gradients/Switch_3Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
c
gradients/Shape_4Shapegradients/Switch_3*
T0*
_output_shapes
:*
out_type0
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
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*1
_output_shapes
:���������@: *
T0*
N
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*
N*
T0*/
_output_shapes
:���������@*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
_output_shapes
:@*
T0*
data_formatNHWC
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
T0*
_output_shapes
:*
out_type0
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
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*&
_output_shapes
:@@
�
Agradients/sequential_1/conv2d_2/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*/
_output_shapes
:���������@*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput*
T0
�
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*&
_output_shapes
:@@*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*
T0*/
_output_shapes
:���������@
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
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
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
_output_shapes
:*
out_type0*
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
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         @   
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
T0*&
_output_shapes
:@*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
beta1_power/initial_valueConst*
_output_shapes
: *
dtype0*
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
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
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
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
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
T
zeros_2Const*
dtype0*
_output_shapes
:@*
valueB@*    
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
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
~
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
T
zeros_3Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_1/bias/Adam_1
VariableV2*
_output_shapes
:@*
dtype0*
shape:@*
	container * 
_class
loc:@conv2d_1/bias*
shared_name 
�
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
l
zeros_4Const*%
valueB@@*    *
dtype0*&
_output_shapes
:@@
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
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
l
zeros_5Const*%
valueB@@*    *&
_output_shapes
:@@*
dtype0
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
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
T
zeros_6Const*
dtype0*
_output_shapes
:@*
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
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
~
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
T
zeros_7Const*
valueB@*    *
dtype0*
_output_shapes
:@
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
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0
b
zeros_8Const*
dtype0*!
_output_shapes
:���* 
valueB���*    
�
dense_1/kernel/Adam
VariableV2*
shared_name *!
_class
loc:@dense_1/kernel*
	container *
shape:���*
dtype0*!
_output_shapes
:���
�
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
b
zeros_9Const*!
_output_shapes
:���*
dtype0* 
valueB���*    
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
zeros_10Const*
_output_shapes	
:�*
dtype0*
valueB�*    
�
dense_1/bias/Adam
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
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
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
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
VariableV2*
shared_name *
shape:	�
*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
dtype0*
	container 
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
valueB	�
*    *
_output_shapes
:	�
*
dtype0
�
dense_2/kernel/Adam_1
VariableV2*
shared_name *
shape:	�
*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
dtype0*
	container 
�
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1zeros_13*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0*
validate_shape(*
use_locking(
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
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

U
zeros_15Const*
valueB
*    *
_output_shapes
:
*
dtype0
�
dense_2/bias/Adam_1
VariableV2*
shape:
*
_output_shapes
:
*
shared_name *
_class
loc:@dense_2/bias*
dtype0*
	container 
�
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0*
validate_shape(*
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

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w�?
Q
Adam/epsilonConst*
valueB
 *w�+2*
_output_shapes
: *
dtype0
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
use_locking( 
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
use_locking( 
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking( 
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0*
use_locking( 
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
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
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
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
�
AdamNoOp&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: ""V
lossesL
J
"softmax_cross_entropy_loss/value:0
$softmax_cross_entropy_loss_1/value:0"
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0"
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
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0"�
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:080��       ��-	c���ec�A*

lossZ)@{}�       ��-	_\��ec�A*

lossr�	@6�7       ��-	��ec�A*

lossQC@�"�       ��-	Ѱ��ec�A*

loss�<�?ȎHb       ��-	�X��ec�A*

loss&k�?�2       ��-	���ec�A*

loss��?�J�       ��-	ı��ec�A*

loss��~?�=B       ��-	���ec�A*

lossd�~?�Y�K       ��-	�=��ec�A	*

loss@�a?�^4       ��-	�ݨ�ec�A
*

loss��N?��x       ��-	����ec�A*

loss��?׼��       ��-	]2��ec�A*

lossD� ?�X��       ��-	
ܪ�ec�A*

loss�3?��s       ��-	���ec�A*

loss��?���       ��-	� ��ec�A*

loss��=?�6]       ��-	Ȭ�ec�A*

loss��>P��z       ��-	�x��ec�A*

lossr��>���!       ��-	m��ec�A*

loss.�>Q�Ϫ       ��-	s���ec�A*

lossD.?�}Q       ��-	�a��ec�A*

loss�\?SD&(       ��-	����ec�A*

lossLk!?|�SX       ��-	נ��ec�A*

loss��?.˙       ��-	pC��ec�A*

loss-�?T�       ��-	x��ec�A*

loss��?ٟ>,       ��-	ڌ��ec�A*

loss8��>;co�       ��-	x)��ec�A*

lossd�>���       ��-	eó�ec�A*

lossD�>UE(�       ��-	�t��ec�A*

loss:��>4r�       ��-	C��ec�A*

loss��>���       ��-	-���ec�A*

loss��?���       ��-	�׶�ec�A*

loss ��>Ũ+�       ��-	Gt��ec�A *

loss�?�.b�       ��-	�0��ec�A!*

loss��?u6��       ��-	dʸ�ec�A"*

loss��>�tg       ��-	=d��ec�A#*

loss�q?��D       ��-	���ec�A$*

loss�>�>Cs�       ��-	X���ec�A%*

loss[?��O�       ��-	M��ec�A&*

lossO�?��/�       ��-	$��ec�A'*

loss?�I�       ��-	ܝ��ec�A(*

loss�L�>�_��       ��-	=ֽ�ec�A)*

loss���>�"�7       ��-	���ec�A**

lossn��>��po       ��-	�G��ec�A+*

loss&y�>�G�       ��-	a��ec�A,*

loss�Ԝ><�K3       ��-	���ec�A-*

lossa�?�r)�       ��-	����ec�A.*

loss]��>v�o�       ��-	�W��ec�A/*

loss���>��*       ��-	����ec�A0*

loss��>�{5�       ��-	���ec�A1*

loss��>�E�       ��-	���ec�A2*

lossd~<>j��       ��-	����ec�A3*

loss(H�>�}]�       ��-	�k��ec�A4*

lossa��>=۽       ��-	���ec�A5*

loss׎>?d7/y       ��-	����ec�A6*

loss%+>o!�{       ��-	3��ec�A7*

loss_>y��       ��-	����ec�A8*

lossZ>��zg       ��-	h��ec�A9*

losss�C>"       ��-	���ec�A:*

loss�'�>� ��       ��-	���ec�A;*

loss >��I�       ��-	]5��ec�A<*

lossڹ�>*%��       ��-	 ���ec�A=*

loss�XZ>��       ��-	�o��ec�A>*

loss�<u>��Vq       ��-	9	��ec�A?*

loss��>���       ��-	���ec�A@*

loss[�=>[�Ů       ��-	K;��ec�AA*

loss�u>�p��       ��-	����ec�AB*

loss��>R&G       ��-	���ec�AC*

loss�V�>?��1       ��-	�5��ec�AD*

loss��>�SWo       ��-	����ec�AE*

loss�Bn>��#�       ��-	����ec�AF*

loss��E>���       ��-	�*��ec�AG*

loss�ə>�kG       ��-	l���ec�AH*

loss�Q>��sA       ��-	Gq��ec�AI*

loss�>�>���       ��-	���ec�AJ*

loss;T�>h       ��-	����ec�AK*

loss�;�>S��&       ��-	C��ec�AL*

loss�� ? 0�       ��-	����ec�AM*

loss�\?"HL       ��-	�{��ec�AN*

loss��Z>U"c�       ��-	���ec�AO*

lossv8>�H       ��-	"���ec�AP*

loss��>P��O       ��-	Z��ec�AQ*

loss���>��       ��-	���ec�AR*

lossڌ�>}v�       ��-	���ec�AS*

loss�;>^��       ��-	�N��ec�AT*

loss�7�>!�       ��-	���ec�AU*

loss��G>����       ��-	]���ec�AV*

loss/�S>�+��       ��-	�!��ec�AW*

loss���>ο��       ��-	r���ec�AX*

loss�^�>�hN�       ��-	�V��ec�AY*

lossE�>����       ��-	����ec�AZ*

loss6��>�^/�       ��-	f���ec�A[*

loss�%?>0���       ��-	���ec�A\*

lossq�>�f�p       ��-	���ec�A]*

loss��>40e�       ��-	R��ec�A^*

loss�V�>lv`�       ��-	����ec�A_*

loss���=,�b�       ��-	��ec�A`*

loss��]>���'       ��-	���ec�Aa*

loss_%i>�T:�       ��-	����ec�Ab*

loss�I�>��(-       ��-	����ec�Ac*

loss��a>>�w\       ��-	p��ec�Ad*

loss�g�=�wF
       ��-	���ec�Ae*

loss2�=�(�7       ��-	d��ec�Af*

loss���>��~�       ��-	.9��ec�Ag*

lossg�#>�p       ��-	�B��ec�Ah*

lossMi)>�3�9       ��-	�v��ec�Ai*

lossxH@>ґ�       ��-	&m��ec�Aj*

loss�8�=���       ��-	^��ec�Ak*

loss8��=�ײc       ��-	EG��ec�Al*

loss��?/�(       ��-	�:��ec�Am*

loss�Y|>�}Z       ��-	���ec�An*

loss�ت>�`(       ��-	�,��ec�Ao*

loss:S/>Gݾ�       ��-	�n��ec�Ap*

loss�u>Z�O	       ��-	���ec�Aq*

loss�s�=�[a       ��-	�	��ec�Ar*

loss� >�U�       ��-	l��ec�As*

loss�o�=s"@�       ��-	�<��ec�At*

loss_X>$%q�       ��-	�?��ec�Au*

loss?�y>�?��       ��-	K9��ec�Av*

lossh��>j�O       ��-	����ec�Aw*

loss���=�6�       ��-	����ec�Ax*

loss}%b>i�       ��-	�v��ec�Ay*

loss�.X>D�       ��-	�3��ec�Az*

loss�PN>'��'       ��-	]���ec�A{*

loss�VB>L[Ta       ��-	@���ec�A|*

lossOa�=7"�O       ��-	-C��ec�A}*

loss>�I[       ��-	����ec�A~*

loss��W>��7       ��-	���ec�A*

loss G�=�x�       �	�9��ec�A�*

loss�~j>d�fI       �	���ec�A�*

loss7cd>�U	/       �	����ec�A�*

lossMp3>�e6�       �	(���ec�A�*

loss�Dw>:���       �	�G �ec�A�*

loss�^�=P�
e       �	F� �ec�A�*

loss:\�=��^       �	��ec�A�*

loss{>.�@7       �	��ec�A�*

lossi�i>[��r       �	���ec�A�*

loss�J>%:       �	�ec�A�*

loss�_>p&<       �	���ec�A�*

loss���=�X��       �	 F�ec�A�*

loss���=Y"��       �	���ec�A�*

loss,h=��2w       �	�}�ec�A�*

loss&8>��n�       �	�;	�ec�A�*

lossۣ>y��       �	I�	�ec�A�*

lossz	>��       �	�u
�ec�A�*

lossڙ�>��k�       �	��ec�A�*

loss���=C�4       �	_��ec�A�*

lossaA7>����       �	�\�ec�A�*

loss�Sa=�8�       �	��ec�A�*

loss���=�i�S       �	9��ec�A�*

loss�n>�� �       �	NG�ec�A�*

loss:0>�=e�       �	���ec�A�*

lossJ�>�l�]       �	Q��ec�A�*

loss�@�=^��b       �	��ec�A�*

loss���>����       �	��ec�A�*

loss�>H�D�       �	�R�ec�A�*

loss
�=B;H�       �	j��ec�A�*

loss���=v�5       �	B��ec�A�*

loss��u>�!b"       �	/�ec�A�*

loss��B>���2       �	?��ec�A�*

loss1�8>"�z       �	�B�ec�A�*

losswK>�v�       �	*:�ec�A�*

loss��>6�]       �	���ec�A�*

lossS>��-A       �	~�ec�A�*

loss1�=U�$       �	�ec�A�*

loss��$>����       �	���ec�A�*

loss���=Gk!       �	�k�ec�A�*

loss
CV=���       �	��ec�A�*

loss���=��       �	���ec�A�*

loss��>���       �	V�ec�A�*

loss��0>^r�b       �	���ec�A�*

loss��=�Ɓ       �	l��ec�A�*

loss��<Y>�       �	NF�ec�A�*

loss��o>��Mk       �	���ec�A�*

loss(��>K-�       �	J��ec�A�*

loss�M�=�y��       �	�K�ec�A�*

lossŋ>
��       �	C �ec�A�*

loss,�>���       �	!�ec�A�*

lossў/>Pڥu       �	̚!�ec�A�*

loss�=H�-l       �	��"�ec�A�*

lossJ��=P��-       �	��#�ec�A�*

loss�2�=�:�       �	��$�ec�A�*

loss��2>�:��       �	�\%�ec�A�*

loss�[=�o�8       �	O&�ec�A�*

loss�<a=�Q       �	R�&�ec�A�*

loss�>7���       �	I�'�ec�A�*

loss�A>�S�z       �	~�(�ec�A�*

loss��=T9�y       �	s�)�ec�A�*

lossJ�>��       �	 W+�ec�A�*

lossOS>0[       �	��+�ec�A�*

loss�u�=.|       �	��,�ec�A�*

loss�u?>2%�J       �	/�-�ec�A�*

loss�(�=�op       �	IL.�ec�A�*

loss}��=%B�       �	x�.�ec�A�*

loss��1>v�I       �	�/�ec�A�*

loss�]=���       �	C70�ec�A�*

lossȲ=P���       �	^1�ec�A�*

lossQZ�>ރ��       �	~�1�ec�A�*

lossѬ>�-��       �	1�2�ec�A�*

loss�ˀ>��|       �	n3�ec�A�*

loss_o�=�M)3       �	�4�ec�A�*

loss��O>��       �	Ϡ4�ec�A�*

lossn=�=�@F�       �	X<5�ec�A�*

loss�K>f��       �	�5�ec�A�*

lossa�>�ą�       �	{�6�ec�A�*

lossù]>����       �		7�ec�A�*

loss��4>Ed��       �	��7�ec�A�*

loss���=%fk       �	�[8�ec�A�*

loss���=��O       �	�9�ec�A�*

loss��!>�,F       �	�9�ec�A�*

loss�c>uދ�       �	�>:�ec�A�*

lossC/>>�4FS       �	��:�ec�A�*

loss5�>� e�       �	��;�ec�A�*

loss�l@=�t��       �	H�<�ec�A�*

lossX�= P:       �	�g=�ec�A�*

lossE��=����       �	8>�ec�A�*

loss��[>L��Z       �	��>�ec�A�*

loss,!>��ľ       �	�b?�ec�A�*

loss�-U>#3&�       �	�@�ec�A�*

loss��!>3��<       �	��@�ec�A�*

loss̸M>Lb�       �	MMA�ec�A�*

lossX	F=�w=.       �	��A�ec�A�*

loss�7�=]�       �	��B�ec�A�*

lossi(�>=�S�       �	�5C�ec�A�*

lossI�4>���_       �	��C�ec�A�*

lossVƘ=.�       �	ǀD�ec�A�*

loss9~=g��       �	qE�ec�A�*

lossEff=HU�       �	LF�ec�A�*

losslPC>%OZ�       �	U�F�ec�A�*

loss�x >d%u-       �	rkG�ec�A�*

loss�>��
�       �	TH�ec�A�*

lossT��=K`A       �	V�H�ec�A�*

losso�<�MB       �	whI�ec�A�*

loss�_>����       �	#J�ec�A�*

loss��X>��P�       �	p�J�ec�A�*

loss��>�ou�       �	�UK�ec�A�*

loss��B>��0I       �	��K�ec�A�*

loss��>�+�X       �	��L�ec�A�*

loss�h�>ױ�       �	�5M�ec�A�*

loss,!>�9�       �	|�M�ec�A�*

loss���=�F��       �	emN�ec�A�*

loss6)�=S,d�       �	�O�ec�A�*

loss��=���       �	ްO�ec�A�*

lossW1->��       �	+KP�ec�A�*

loss�Q>��u       �	P�P�ec�A�*

loss �>�h��       �	x{Q�ec�A�*

loss���=CAZ       �	u!R�ec�A�*

lossX��=�m��       �	��R�ec�A�*

loss֛>Ȯ�y       �	fkS�ec�A�*

lossl >����       �	�T�ec�A�*

lossL�>+W       �	��T�ec�A�*

losse��=b_ބ       �	ۈU�ec�A�*

loss��~>1l��       �	T5V�ec�A�*

lossC�
>� 
�       �	��V�ec�A�*

loss{"�=M��       �	9�W�ec�A�*

loss�}�=��0�       �	\9X�ec�A�*

lossu=>�QJ       �		�X�ec�A�*

loss�E,>�]N       �	ۈY�ec�A�*

loss��~>T�e       �	�*Z�ec�A�*

loss��=����       �	&�Z�ec�A�*

loss�B>qS�_       �	!�[�ec�A�*

loss��>ޡDa       �	=\�ec�A�*

loss��\=�p       �	��\�ec�A�*

loss�$,>L'=�       �	4�]�ec�A�*

loss�Ü=���       �	'1^�ec�A�*

loss?1R>e��       �	�_�ec�A�*

loss�?�=Xx       �	F�_�ec�A�*

loss��.>g�       �	@�`�ec�A�*

loss �1>~@�       �	�:a�ec�A�*

loss׽�>�5E       �	��a�ec�A�*

loss��>L�v�       �	�c�ec�A�*

loss���=9l�y       �	k�c�ec�A�*

loss�b�=���       �	:�d�ec�A�*

loss���=�d       �	��e�ec�A�*

loss`4�=
oT�       �	yvf�ec�A�*

loss��>>oi�       �	�g�ec�A�*

loss�H_>L��       �	��g�ec�A�*

loss�Ju=� E       �	�Ih�ec�A�*

loss� &=Ѫ�       �	��h�ec�A�*

loss��G>��!�       �	 �i�ec�A�*

lossKk>���0       �	H4j�ec�A�*

loss?-=&?�       �	��k�ec�A�*

loss;��=]_U       �	)�l�ec�A�*

loss�/R>%,@�       �	/3m�ec�A�*

loss�,�=]�T       �	��m�ec�A�*

lossg>���       �	in�ec�A�*

loss)e>�QV       �	�o�ec�A�*

loss9�=���I       �	Qp�ec�A�*

lossz��=���       �	�p�ec�A�*

lossS��=j�cv       �	��q�ec�A�*

loss�p=V��       �	�@r�ec�A�*

loss�7S=�4�       �	#�r�ec�A�*

loss.��=1�&�       �	�vs�ec�A�*

loss	��<B:~_       �	�t�ec�A�*

loss;��<��       �	��t�ec�A�*

loss��<i���       �	5Cu�ec�A�*

loss�!�=~d�q       �	��u�ec�A�*

lossy�>$�w]       �	d�v�ec�A�*

loss3c>��Jj       �	�8w�ec�A�*

loss�Y�>{��       �	(�w�ec�A�*

loss��>��       �	�rx�ec�A�*

lossm>�D!|       �	zy�ec�A�*

loss�0\=Z&�       �	^�y�ec�A�*

lossP�>Y	       �	/lz�ec�A�*

loss
�>(�       �	�{�ec�A�*

loss̈́�=��<�       �	Ҩ{�ec�A�*

loss%n]>�ȧ       �	�G|�ec�A�*

loss��=�+R       �	\�|�ec�A�*

lossGT>i�       �	(|}�ec�A�*

lossc�6=�R�5       �	g&~�ec�A�*

lossC[>+ ��       �	��~�ec�A�*

loss}>>0�Af       �	__�ec�A�*

loss��>�Q#       �	L��ec�A�*

loss��>b���       �	ݖ��ec�A�*

loss���=�`�G       �	�,��ec�A�*

lossZ(�=54G       �	�΁�ec�A�*

loss��
>e�n       �	wi��ec�A�*

loss4m�= �v       �	��ec�A�*

loss�X=��'�       �	����ec�A�*

lossDa�=���m       �	�-��ec�A�*

loss��*>YC�       �	�˄�ec�A�*

loss�`=�/ $       �	�g��ec�A�*

loss�1>�
��       �	���ec�A�*

loss�<>[�       �	����ec�A�*

loss#TU=�S       �	�6��ec�A�*

loss�O3>U�^       �	�݇�ec�A�*

loss�H>0� =       �	�~��ec�A�*

loss;#>KeD       �	���ec�A�*

lossR��=�s�Y       �	���ec�A�*

loss�T>��       �	�Q��ec�A�*

loss�s>$9�       �	u��ec�A�*

loss��=Tx̒       �	l|��ec�A�*

loss;��=�df       �	���ec�A�*

loss��=`�       �	��ec�A�*

loss��N=Ι�       �	b��ec�A�*

lossR4*>��Ǵ       �	����ec�A�*

loss���=�8RR       �	p���ec�A�*

loss��}=d��       �	�M��ec�A�*

loss(ٜ>A�V       �	���ec�A�*

loss3�=e��       �	����ec�A�*

loss�h=a�(3       �	�/��ec�A�*

loss�׏=�r	       �	,֑�ec�A�*

loss:��=�R       �	~r��ec�A�*

loss��=��z8       �	���ec�A�*

loss���=�u
�       �	����ec�A�*

loss��>s�       �	\U��ec�A�*

loss{>.;~	       �	x��ec�A�*

lossn�=�K�       �	R���ec�A�*

loss3a>>>hd]       �	{1��ec�A�*

loss��>�튑       �	Ж�ec�A�*

lossĭ>�-�       �	 r��ec�A�*

loss,�>P��)       �	
��ec�A�*

lossi�v=>�W       �	���ec�A�*

lossiV�=�m'       �	�\��ec�A�*

loss���=3 �       �	����ec�A�*

loss<d>wO�       �	��ec�A�*

loss��=���D       �	�6��ec�A�*

loss�Ȉ=�ʽ�       �	oכ�ec�A�*

lossQ;�=_R�       �	�|��ec�A�*

loss���=���       �	
���ec�A�*

loss�K�=J���       �	#,��ec�A�*

loss��=��       �	LǞ�ec�A�*

loss�<�=��x+       �	����ec�A�*

loss���=�C��       �	�"��ec�A�*

losse�t>vsL       �	����ec�A�*

lossq��=!�S�       �	d��ec�A�*

lossּ>�y�       �	����ec�A�*

loss'3>�a       �	�H��ec�A�*

loss:nl=n��c       �	u:��ec�A�*

loss�h=;�       �	O>��ec�A�*

loss�>q)�       �	|֥�ec�A�*

loss�M�=�        �	�p��ec�A�*

loss�ô=��&        �	���ec�A�*

loss��=7���       �	S���ec�A�*

loss7��=�Y��       �	~T��ec�A�*

loss�}�=cS�       �	���ec�A�*

loss�r>���       �	M���ec�A�*

lossW>����       �	�?��ec�A�*

loss ��=����       �	�ߪ�ec�A�*

loss�f=��9�       �	���ec�A�*

loss�v[>�Z       �	� ��ec�A�*

lossy�=����       �	H��ec�A�*

loss!� >~dh�       �	ǀ��ec�A�*

loss�>��y       �	��ec�A�*

loss
L=���       �	ع��ec�A�*

lossWe�<�^��       �	p_��ec�A�*

lossce�=�Z       �		��ec�A�*

loss��>D�       �	`���ec�A�*

lossB>loY�       �	$G��ec�A�*

loss�*�=�as�       �	���ec�A�*

loss�#C>=h
�       �	���ec�A�*

loss��>*�       �	� ��ec�A�*

loss��Y=ԿU�       �	U���ec�A�*

loss�,�=4�=�       �	&V��ec�A�*

loss���=P��$       �	����ec�A�*

loss�`�=`��       �	P���ec�A�*

loss�V>�Iͦ       �	�,��ec�A�*

loss!R�<Z|�       �	�Ӷ�ec�A�*

lossbe=��=       �	�u��ec�A�*

loss��>_���       �	{��ec�A�*

lossX��=�~�       �	��ec�A�*

loss�p5=%�On       �	3R��ec�A�*

loss7߿=�螎       �	p��ec�A�*

loss��e>���<       �	���ec�A�*

lossLÙ<�"�       �	JB��ec�A�*

loss;��=dڶJ       �	���ec�A�*

loss��>xk*|       �	���ec�A�*

loss�s�>�3Z       �	�$��ec�A�*

loss&�>S�[�       �	TȽ�ec�A�*

loss�!�=�(-       �	i��ec�A�*

loss{Ha>z���       �	|��ec�A�*

loss-�>}_�       �	İ��ec�A�*

loss�hr>i�m�       �	�P��ec�A�*

losse��=�P       �	@���ec�A�*

lossi��>AN̔       �	Ú��ec�A�*

loss��5>~���       �	9��ec�A�*

loss��=�BE�       �	����ec�A�*

loss�|=
��.       �	����ec�A�*

loss���=�ۈ�       �	�"��ec�A�*

losstGx>�g,�       �	Ͻ��ec�A�*

loss�C>�l       �	g��ec�A�*

loss�c�=�4�       �	���ec�A�*

loss�)>�� �       �	ݙ��ec�A�*

loss�%r=Te�a       �	�1��ec�A�*

lossI�g>$��       �	h���ec�A�*

loss�f>�\�       �	4i��ec�A�*

loss�}>��]�       �	����ec�A�*

lossG>�Qj       �	B���ec�A�*

loss�kB>�oX�       �	�2��ec�A�*

loss��=�@�C       �	C���ec�A�*

loss�I}=���       �	ob��ec�A�*

loss*&�=lz��       �	L���ec�A�*

loss470>#r�1       �	o���ec�A�*

loss�j4=C��       �	 ;��ec�A�*

loss�y�=��pM       �		���ec�A�*

loss��>����       �	ۅ��ec�A�*

lossI�Z=�        �	@/��ec�A�*

loss��^=�8�       �	����ec�A�*

loss��=?c�       �	�m��ec�A�*

loss�%=��       �	��ec�A�*

loss`�G=����       �	д��ec�A�*

lossJύ>���       �	�_��ec�A�*

loss�� >.%�d       �	����ec�A�*

lossD�=Dp\       �	ݖ��ec�A�*

loss���=��       �		5��ec�A�*

loss��y=����       �	����ec�A�*

lossM>�9��       �	�g��ec�A�*

loss���>��I�       �	T��ec�A�*

lossn:>o���       �	"���ec�A�*

loss\�{=��Z       �	�E��ec�A�*

loss�#>��	�       �	����ec�A�*

loss��6>g D       �	k���ec�A�*

loss�w�=���J       �	���ec�A�*

lossr,�=�NC�       �	#���ec�A�*

loss��n>��X�       �	�S��ec�A�*

lossFSW>�m��       �	A���ec�A�*

lossc#M>h(q       �	����ec�A�*

loss���=E��Y       �	.��ec�A�*

lossֆ�=���       �	����ec�A�*

loss=թ=���       �	Wz��ec�A�*

lossw��=�6]       �	���ec�A�*

lossΈ�=��"       �	���ec�A�*

loss�S�<�       �	#h��ec�A�*

loss���=!�e�       �	���ec�A�*

loss��.>�Y��       �	����ec�A�*

loss��=z���       �	�F��ec�A�*

lossY&=��+�       �	����ec�A�*

loss~�=�`MG       �	���ec�A�*

loss�t�==���       �	�2��ec�A�*

lossȴ
> �       �	����ec�A�*

loss�p<���       �	���ec�A�*

loss\�:>E�H�       �	��ec�A�*

loss`�=�7�|       �	y���ec�A�*

loss�:�=�o�v       �	2���ec�A�*

loss48�>�+o       �	�5��ec�A�*

lossl��=A/       �	����ec�A�*

loss�^h=��       �	W{��ec�A�*

lossB==�Ӝ       �	7��ec�A�*

lossIC�=�܌z       �	6���ec�A�*

loss��=��       �	RI��ec�A�*

loss��>Z!       �	�(��ec�A�*

lossl��>��|�       �	����ec�A�*

loss���=ƅU       �	�\��ec�A�*

loss�J�=��       �	����ec�A�*

lossd��=a�'�       �	=���ec�A�*

loss���=�s.       �	&:��ec�A�*

lossX�5=n_�       �	N���ec�A�*

loss�)�=���N       �	Eg��ec�A�*

lossT��=>��       �	C��ec�A�*

lossqn=�I{       �	Ϡ��ec�A�*

loss`�'>gW��       �	�9��ec�A�*

lossrt�=X#�;       �	����ec�A�*

loss!�K>��1Q       �	Ct��ec�A�*

loss���<�s�       �	���ec�A�*

loss)/�=ztH�       �	����ec�A�*

loss
T9=	�.�       �	�N��ec�A�*

lossW@5=<�1       �	l���ec�A�*

loss�Ө=u�       �	����ec�A�*

loss�c�<1�C"       �	�&��ec�A�*

loss���<�im)       �	`���ec�A�*

loss3ͻ=KE�       �	�a��ec�A�*

loss\7>�R�       �	����ec�A�*

loss$��<A�       �	{���ec�A�*

loss�!=�t�G       �	�r��ec�A�*

lossw��=�t�       �	E��ec�A�*

loss�ʻ=��H       �	���ec�A�*

lossE�=WJ��       �	�X��ec�A�*

loss���=�'��       �	����ec�A�*

loss\*=&��       �	O���ec�A�*

loss==�=I��       �	�-��ec�A�*

loss�Ӥ=��-       �	����ec�A�*

loss���=!�'�       �	`���ec�A�*

lossd��=��       �	�3��ec�A�*

loss~=�jr�       �	|���ec�A�*

loss-�c=f��       �	]� �ec�A�*

lossQ�=�%K       �	�v�ec�A�*

loss�8>C��n       �	~�ec�A�*

lossL>\B��       �	ù�ec�A�*

loss� >��g�       �	�W�ec�A�*

loss�Eh=��7       �	���ec�A�*

lossc>)�۰       �	��ec�A�*

loss��r=@���       �	�@�ec�A�*

loss6\�=��G�       �	&��ec�A�*

lossid�<D�       �	�~�ec�A�*

lossŪ"=�kh6       �	��ec�A�*

loss���=�ue�       �	J��ec�A�*

loss|%E= [&       �	�M�ec�A�*

loss��=.z��       �	h��ec�A�*

loss|1�=Ү�I       �	Q�	�ec�A�*

lossO>�!�       �	~
�ec�A�*

loss�	�<���       �	��
�ec�A�*

loss��='�k       �	eP�ec�A�*

loss�t>���m       �	���ec�A�*

loss
��=YP�       �	���ec�A�*

loss!��='�n       �	n�ec�A�*

loss�D�<	[��       �	���ec�A�*

loss��<�n\�       �	�F�ec�A�*

loss��<j
�|       �	��ec�A�*

loss���<3�bm       �	s�ec�A�*

lossh�!<�ĕ�       �	��ec�A�*

loss
��=��5D       �	���ec�A�*

loss	fF<bn�s       �	�>�ec�A�*

loss��;(i%�       �	���ec�A�*

lossҷ�;�>#       �	Dk�ec�A�*

lossdc�=�ŷ�       �	q�ec�A�*

loss�+>�&ܥ       �	��ec�A�*

lossdY=���F       �	���ec�A�*

loss�F;;��M�       �	V�ec�A�*

loss��=d3       �	���ec�A�*

loss�\|>ҧr       �	��ec�A�*

lossת�<�KrS       �	+4�ec�A�*

loss=Yu�       �	���ec�A�*

lossJt=�#�%       �	>z�ec�A�*

loss��>���       �	'�ec�A�*

lossvi�=��!S       �	ӽ�ec�A�*

loss�%�=���$       �	Y�ec�A�*

lossJ>�=����       �	v��ec�A�*

loss���=$It<       �	���ec�A�*

loss��>��=�       �	& �ec�A�*

loss[��=,�l�       �	���ec�A�*

loss�=}�7       �	3!�ec�A�*

lossټ>�jG2       �	d�!�ec�A�*

loss��_>�	�@       �	Ŏ"�ec�A�*

loss-��=a���       �	%=#�ec�A�*

lossL��=���       �	��#�ec�A�*

loss�k>B��s       �	@�$�ec�A�*

loss�C�=d7�       �	�&�ec�A�*

loss�{�=LԊ{       �	P�&�ec�A�*

loss��=Z�       �	x�'�ec�A�*

loss�R�=U`.!       �	�m(�ec�A�*

loss��D=h�d       �	$a)�ec�A�*

lossl�a=�<:       �	�t*�ec�A�*

lossK�
>KE��       �	�G+�ec�A�*

lossf��=��w�       �	�$,�ec�A�*

loss͇o=�~       �	<-�ec�A�*

loss�c=�x��       �	��-�ec�A�*

loss�>���       �	�.�ec�A�*

loss���=�`EB       �	B\/�ec�A�*

loss�B>��y       �	�0�ec�A�*

lossS�>�(�q       �	�1�ec�A�*

loss�,�=�	�       �	�2�ec�A�*

loss�>3?��       �	{33�ec�A�*

loss*��=ԑv�       �	�'4�ec�A�*

loss���<i�y       �	��4�ec�A�*

lossWx=1�*z       �	C�5�ec�A�*

loss�t+=���        �	�d6�ec�A�*

loss�o�<F#       �	�P7�ec�A�*

loss��>���       �	�7�ec�A�*

loss�$>�r       �	��8�ec�A�*

loss�v
>�=�       �	F�9�ec�A�*

lossŠ==��r       �	^i:�ec�A�*

loss���=�[��       �	�;�ec�A�*

loss�x�=����       �	��;�ec�A�*

loss��">bJ��       �	4d<�ec�A�*

loss��[=����       �	�=�ec�A�*

loss�.�=,2�       �	ӥ=�ec�A�*

loss:��=R&?       �	k}>�ec�A�*

loss��=��hI       �	�?�ec�A�*

lossn�=jk�       �	��?�ec�A�*

loss���=�$��       �	XT@�ec�A�*

lossͮ�=���,       �	��@�ec�A�*

loss�=mM��       �	a]�ec�A�*

loss��=7W]�       �	��]�ec�A�*

loss��=�K�r       �	�H^�ec�A�*

loss*8^=�f��       �	"�^�ec�A�*

loss�i>xp��       �	qt_�ec�A�*

loss�p1>�B       �	c`�ec�A�*

loss��>3o�       �	��`�ec�A�*

loss�SJ>Ȕ        �	{Ka�ec�A�*

loss�0�=:q��       �	�a�ec�A�*

loss��d>]G�k       �	�}b�ec�A�*

lossEZ�=u�>b       �	c�ec�A�*

loss؄"=q	WA       �	��c�ec�A�*

loss���=���.       �	�Md�ec�A�*

loss!1�=�:�d       �	��d�ec�A�*

lossDW�=zO��       �	E�e�ec�A�*

loss��>IQ�       �	{Jf�ec�A�*

loss=L`Eq       �	i�f�ec�A�*

loss��>��F       �	�xg�ec�A�*

loss�:b=��s�       �	Ih�ec�A�*

lossF��>3�       �	~qi�ec�A�*

loss��=�r�1       �	�j�ec�A�*

loss�N�=�%�       �	�j�ec�A�*

loss��!=A�?       �	a8k�ec�A�*

loss`->�ɤ       �	g�k�ec�A�*

loss��`=�=�K       �	}wl�ec�A�*

lossŸ=Q�m       �	�!m�ec�A�*

loss�>6��"       �	��m�ec�A�*

loss�_b=���K       �	��n�ec�A�*

lossx��=Fe2�       �	x*o�ec�A�*

lossH��=��ّ       �	g�o�ec�A�*

lossdӋ=t&��       �	Sup�ec�A�*

loss���=��?       �	�jq�ec�A�*

loss=��=��Oh       �	r�ec�A�*

losst>�J�'       �	��r�ec�A�*

loss�7 >� �-       �	��s�ec�A�*

loss��>�jܾ       �	�(t�ec�A�*

lossq��<�_��       �	*u�ec�A�*

loss<A�=޹�       �	Z�u�ec�A�*

lossH�5>��R�       �	�<v�ec�A�*

lossF�>�XR       �	��v�ec�A�*

loss�{ >�tn8       �	9�w�ec�A�*

loss�$�=�'o       �	�:x�ec�A�*

loss�<��       �	x�x�ec�A�*

loss4=�=N��       �	�ny�ec�A�*

lossSg�=�9�       �	�z�ec�A�*

lossIC>\r�       �	C�z�ec�A�*

loss��Z=��       �	�P{�ec�A�*

loss��o=��'!       �	_�{�ec�A�*

lossX=���       �	��|�ec�A�*

losss=�Lj�       �	�!}�ec�A�*

loss�[<�	�       �	��}�ec�A�*

loss��=��b�       �	�_~�ec�A�*

loss!��<�sh�       �	�S�ec�A�*

loss6�0?V�P[       �	 ��ec�A�*

loss��P=��a�       �	횀�ec�A�*

loss6()<�4x       �	�=��ec�A�*

lossH�=<���]       �	���ec�A�*

loss���=�8��       �	���ec�A�*

loss��=���       �	�1��ec�A�*

loss��Q=�w�       �	���ec�A�*

lossԆ�=�*�       �	����ec�A�*

loss{Sg=v��9       �	�"��ec�A�*

loss�a�<�Th       �		Ņ�ec�A�*

loss�=�=����       �	�d��ec�A�*

lossw�=�N9       �	���ec�A�*

loss�i<\bA�       �	���ec�A�*

loss���= ��       �	*;��ec�A�*

lossC>Nb�       �	�Ո�ec�A�*

loss�t$>�	a       �	(��ec�A�*

loss��#>ꏦ�       �	�+��ec�A�*

loss1S�=2�Y�       �	�Ê�ec�A�*

loss�i=��<�       �	sc��ec�A�*

lossZHi=�^�       �	���ec�A�*

loss���=R�%       �	����ec�A�*

loss��=7�d>       �	�@��ec�A�*

losscN�<�l�)       �	�ڍ�ec�A�*

lossH��=�}i�       �	hu��ec�A�*

loss�n�=��       �	���ec�A�*

loss�)w=����       �	���ec�A�*

lossE��=�Wϕ       �	�Q��ec�A�*

loss_`=�\��       �	L���ec�A�*

loss��=4H�2       �	4���ec�A�*

loss㼵=��&�       �	�_��ec�A�*

lossZ�=��       �	J	��ec�A�*

loss�zg=�"?m       �	8���ec�A�*

lossd� =J�ٶ       �	^G��ec�A�*

loss��=�pB       �	���ec�A�*

loss�P�<n\e�       �	f���ec�A�*

lossq�q=3��b       �	r1��ec�A�*

loss��l=�'�       �	�і�ec�A�*

lossZ�i=)i�       �	�j��ec�A�*

lossIF�=	�V�       �	�#��ec�A�*

loss�%=z3lh       �	����ec�A�*

loss���=\�)       �	�U��ec�A�*

loss
�>��**       �	���ec�A�*

lossl�=�A,,       �	9���ec�A�*

loss�j�=�+�       �	�4��ec�A�*

loss� ,=�O#       �	�̛�ec�A�*

lossx�6>���:       �	�i��ec�A�*

lossF >�ZB       �	G��ec�A�*

loss�&_=7^lR       �	ؚ��ec�A�*

lossګ�<�h��       �	1��ec�A�*

loss���=��pj       �	�ƞ�ec�A�*

loss�=`�9C       �	@l��ec�A�*

loss�^=�Ii�       �	K��ec�A�*

loss.�d=D�G�       �	e���ec�A�*

loss��<-�       �	�D��ec�A�*

lossK�=��'       �	Lޡ�ec�A�*

loss���=�ƅu       �	�s��ec�A�*

loss��=z���       �	��ec�A�*

loss�Z�=`��       �	����ec�A�*

loss�=���       �	M��ec�A�*

loss�(=��њ       �	s��ec�A�*

losss��</���       �	�9��ec�A�*

lossn�=�]�       �	Rզ�ec�A�*

loss�=4�"�       �	aq��ec�A�*

loss�=C=�M<       �	�)��ec�A�*

loss�5�=�R�       �	�Ĩ�ec�A�*

loss���=�j(�       �	�`��ec�A�*

loss���=�Pp       �	����ec�A�*

loss��=��t       �	����ec�A�*

lossq�N=�vr       �	�C��ec�A�*

loss�<�=B�f�       �	��ec�A�*

loss#�Q=�M��       �	�{��ec�A�*

loss�`1<�~       �	�-��ec�A�*

loss�i�=���       �	�׭�ec�A�*

lossj'�=�Ն       �	u��ec�A�*

loss&ז=�I�       �	���ec�A�*

loss��/>An9n       �	?���ec�A�*

loss�>A\	�       �	,G��ec�A�*

lossZ��=�!F       �	L��ec�A�*

loss�"=�1�U       �	����ec�A�*

loss���=< �       �	�N��ec�A�*

loss�߹<��Tx       �	)��ec�A�*

lossO��=&�v       �	��ec�A�*

loss�>�=��J�       �	v3��ec�A�*

loss��f=O�A       �	Eֵ�ec�A�*

loss1wO=ԸP�       �	9|��ec�A�*

loss�_=/���       �	���ec�A�*

loss���<�7Y&       �	a���ec�A�*

loss�FW<eb��       �	�_��ec�A�*

loss�=DBU�       �	vl��ec�A�*

loss�~>�nd�       �	�
��ec�A�*

loss�)>ݷ�        �	����ec�A�*

loss��x>Xӱw       �	�>��ec�A�*

loss�.Q=��[       �	
-��ec�A�*

loss�N�=aO�        �	RӼ�ec�A�*

loss��==��j       �	Z���ec�A�*

lossO�Q=�B+X       �	�*��ec�A�*

loss)��=X��3       �	�ʾ�ec�A�*

loss�=mG�       �	޿�ec�A�*

lossE�=�N       �	����ec�A�*

loss�6=�7�       �	�)��ec�A�*

lossRa>�I�N       �	���ec�A�*

lossx��=^���       �	���ec�A�*

loss1��<73r       �	�?��ec�A�*

loss���=s�:       �	����ec�A�*

lossO��=u[��       �	̚��ec�A�*

loss�={70'       �	
K��ec�A�*

loss͂�<�#�       �	���ec�A�*

loss�P�=����       �	���ec�A�*

loss}!=��       �	�9��ec�A�*

loss)� =
���       �	���ec�A�*

lossE��<�K�       �	�z��ec�A�*

loss�=Y��V       �	]��ec�A�*

lossp}=s�a       �	����ec�A�*

lossh�=��^�       �	%\��ec�A�*

loss��>�d��       �	���ec�A�*

lossJ'�<�CRg       �	)���ec�A�*

loss �b=#|�       �	�5��ec�A�*

loss|��=�*�       �	����ec�A�*

loss6�=<�� �       �	M���ec�A�*

loss��F=��v�       �	���ec�A�*

loss%Pv=~m��       �	����ec�A�*

losslD=&�       �	4g��ec�A�*

lossj�=
�.       �	e���ec�A�*

lossj~�=|��        �	����ec�A�*

loss�'i=�]P       �	�?��ec�A�*

loss��=�W��       �	����ec�A�*

loss�<Wu�       �	��ec�A�*

loss�-;q��       �	�4��ec�A�*

lossW5=��ʁ       �	���ec�A�*

loss�<T�o       �	����ec�A�*

loss��<F�       �	�E��ec�A�*

loss�a�;��8       �	����ec�A�*

loss��<��a       �	o���ec�A�*

loss-�!=���L       �	(D��ec�A�*

loss�~{=�b       �	E���ec�A�*

loss# =n���       �	����ec�A�*

loss;>;�9�       �	�D��ec�A�*

loss�g>��       �	]���ec�A�*

lossX�.=���
       �	����ec�A�*

loss�WV<
a�       �	hA��ec�A�*

loss1c= 춶       �	g���ec�A�*

loss�C<����       �	���ec�A�*

loss̞i=����       �	u;��ec�A�*

loss�P=���       �	B���ec�A�*

loss�Č=W       �	����ec�A�*

loss�=8>db       �	Bx��ec�A�*

loss:\=��[1       �	���ec�A�*

loss��+=�H�       �	w���ec�A�*

lossa�<�^�       �	�_��ec�A�*

loss��=Ĳ@�       �	;���ec�A�*

loss�|�=��       �	z���ec�A�*

loss=X =�D^       �	�^��ec�A�*

loss?=|h       �	_��ec�A�*

lossŜ�=NhQ�       �	����ec�A�*

lossq�=���       �	����ec�A�*

loss���=C��       �	���ec�A�*

loss��=Xy��       �	����ec�A�*

lossم=�/�       �	���ec�A�*

loss=�M=�r��       �	���ec�A�*

loss���<. �       �	����ec�A�*

loss��j=�炾       �	����ec�A�*

loss\��=0�Q�       �	���ec�A�*

lossil�=�k��       �	����ec�A�*

loss6!*>L�d       �	DR��ec�A�*

lossr �>)�       �	����ec�A�*

lossT�6=3��       �	����ec�A�*

loss�Ȃ>ċ!       �	"7��ec�A�*

loss6�=�K��       �	����ec�A�*

loss���<˃z%       �	�|��ec�A�*

loss$=�"�       �	"��ec�A�*

loss$F�=}��m       �	����ec�A�*

loss�i%=�p�       �	1[��ec�A�*

loss�3�=��9�       �	����ec�A�*

loss*v�=�+�       �	���ec�A�*

loss5� >��C�       �	�<��ec�A�*

loss%)C=gts]       �	����ec�A�*

lossJ��=HӰT       �	���ec�A�*

loss�Ġ<�|�^       �	&��ec�A�*

loss<'=Y�vD       �	T���ec�A�*

lossw��=u:Z�       �	�f��ec�A�*

loss��&=�:�       �	���ec�A�*

loss��=m�ʋ       �	����ec�A�*

lossZ��=�ܶC       �	WC��ec�A�*

loss��d=�+�i       �	����ec�A�*

loss�>��I�       �	���ec�A�*

loss߬>`45f       �	�8��ec�A�*

loss�#�<�	�       �	����ec�A�*

loss���<��Q       �	~r��ec�A�*

lossȎS=1��       �	*��ec�A�*

loss���=h%�I       �	���ec�A�*

lossB?�=A�%�       �	uX��ec�A�*

loss��=uH<       �	�G �ec�A�*

loss��<:}�V       �	�>�ec�A�*

lossb�=*�ڇ       �	8��ec�A�*

lossR��=��eX       �	���ec�A�*

lossŐ�=8p��       �	R(�ec�A�*

loss3T�<���       �	��ec�A�*

loss��U=+*�       �	 p�ec�A�*

loss-��=��h       �	��ec�A�*

loss�\&=���       �	Q��ec�A�*

loss1�;���       �	>�ec�A�*

loss.�2<���p       �	��ec�A�*

loss;�G<�迍       �	���ec�A�*

loss�=o��       �	k�ec�A�*

loss�_�= ���       �	�	�ec�A�*

loss�2�=�r̕       �	��	�ec�A�*

loss���=p9�P       �	S
�ec�A�*

lossF8�=━�       �	$�
�ec�A�*

loss�TF=b�۹       �	��ec�A�*

lossOL>B|�       �	�1�ec�A�*

loss���=am;'       �	���ec�A�*

lossC�=�W��       �	�h�ec�A�*

lossn&<̾       �	��ec�A�*

lossؔ�=t���       �	Ͻ�ec�A�*

loss�_�=��       �	�^�ec�A�*

loss���=�'�0       �	 �ec�A�*

lossx�=�
��       �	ӣ�ec�A�*

losssn�=w�{       �	�C�ec�A�*

loss�&J=�ş�       �	��ec�A�*

loss�m�=Yi��       �	���ec�A�*

loss�:=�N��       �	c*�ec�A�*

loss$�>�Za       �	���ec�A�*

loss3� >h"�u       �	q�ec�A�*

loss��A=Iaؐ       �	���ec�A�*

loss{5=�
]       �	o��ec�A�*

loss�g�=��       �	�$�ec�A�*

lossϐ�='c�       �	h$�ec�A�*

loss�a�;�ޜ�       �	���ec�A�*

loss�ol=:n	S       �	��ec�A�*

loss�%=�,�H       �	. �ec�A�*

loss�`�<�,*       �	2��ec�A�*

loss�ya=�4�       �	�T�ec�A�*

loss=#=I"�(       �	���ec�A�*

loss���=NF+�       �	��ec�A�*

loss1�)<&���       �	/6�ec�A�*

loss�ܒ=��m~       �	b��ec�A�*

loss�<�;�       �	�n �ec�A�*

loss���<|k��       �	�!�ec�A�*

loss$�=�Hn�       �	+�!�ec�A�*

loss��;�/       �	>["�ec�A�*

loss:�<.��       �	��"�ec�A�*

lossw�!=�z��       �	��#�ec�A�*

loss���=Ԭ�<       �	y<$�ec�A�*

lossc�V=p㖄       �	��$�ec�A�*

loss��j>\/I�       �	�t%�ec�A�*

loss��=gf�       �	�&�ec�A�*

loss
��=Z���       �	��&�ec�A�*

loss�Ԡ=�t �       �	3�'�ec�A�*

loss��p= U5       �	��(�ec�A�*

loss)�|<��I[       �	��)�ec�A�*

lossC��=t�So       �	��*�ec�A�*

loss^�<ֱX�       �	�,�ec�A�*

loss/P>��a%       �	��,�ec�A�*

loss	Or;��       �	3S-�ec�A�*

loss���=A��       �	XT.�ec�A�*

lossz�=�,       �	H/�ec�A�*

loss2��<�xa       �	�g0�ec�A�*

loss�"P=���^       �	K�1�ec�A�*

loss��=�染       �	�z2�ec�A�*

loss\!�=.EL�       �	�I3�ec�A�*

lossc��<�;�       �	��4�ec�A�*

loss�P=�
��       �	�5�ec�A�*

loss��=[��       �	S�6�ec�A�*

lossM�<�rNj       �	��7�ec�A�*

lossrԽ=���       �	�(8�ec�A�*

lossu�=���       �	P�8�ec�A�*

loss���<?���       �	�9�ec�A�*

loss��]<i�A8       �	�L:�ec�A�*

lossQP+>W��       �	��:�ec�A�*

loss���=\�s�       �	2 <�ec�A�*

losssե</�[�       �	E�<�ec�A�*

lossdȯ=���       �	�)>�ec�A�*

loss��`>����       �	E�>�ec�A�*

lossF6�=Z��       �	ڏ?�ec�A�*

loss��i=q���       �	�h@�ec�A�*

lossr�>s�:       �	�MA�ec�A�*

loss�>�N�C       �	��A�ec�A�*

loss��K=@zN�       �	H�B�ec�A�*

loss�~C=\T=�       �	�C�ec�A�*

loss�9�<Ė��       �	m�D�ec�A�*

loss�e.=���       �	��E�ec�A�*

loss�T>���       �	f�F�ec�A�*

loss�`�=T�f�       �	�G�ec�A�*

loss��e=�8G�       �	5�H�ec�A�*

loss�l-=�M       �	>�I�ec�A�*

loss��'=��$I       �	��J�ec�A�*

loss��<�S�#       �	N�K�ec�A�*

lossd�3<Bٙ�       �	�`L�ec�A�*

loss8�&=0��       �	�EM�ec�A�*

loss=*�=G�J�       �	��M�ec�A�*

loss��=E��       �	�N�ec�A�*

lossѡ_>�1[�       �	��O�ec�A�*

loss��=�Uj4       �	�yP�ec�A�*

loss���<��^�       �	8-Q�ec�A�*

loss/�o=���       �	T�Q�ec�A�*

loss�v=r���       �	��R�ec�A�*

loss`�,=�GBU       �	��S�ec�A�*

loss�c=�8�       �	U�ec�A�*

loss,R=��       �	��U�ec�A�*

loss�¡=;�4       �	�V�ec�A�*

loss�h�=��yW       �	s�W�ec�A�*

loss��=[��       �	�AX�ec�A�*

loss�c=���       �	�X�ec�A�*

loss%�W=Q$�       �	[�Y�ec�A�*

loss3=nm��       �	�vZ�ec�A�*

loss}}g<�9(       �	xz[�ec�A�*

lossOSp=n��9       �	�\�ec�A�*

loss��S=�
�       �	��\�ec�A�*

loss�<&�S       �	EJ]�ec�A�*

loss�g�=O�8       �	3�]�ec�A�*

loss��>�*R�       �	J~^�ec�A�*

loss�8>�=e�       �	 _�ec�A�*

loss���=l���       �	ݵ_�ec�A�*

lossx�=R7�       �	�J`�ec�A�*

loss��|=TЦ       �	.�`�ec�A�*

lossava=l���       �	B|a�ec�A�*

losssZ=
��       �	�!b�ec�A�*

loss�U=S��       �	�b�ec�A�*

loss�f=�
�\       �	�dc�ec�A�*

loss݊T=u�r�       �	�	d�ec�A�*

lossn��=5AY       �	��d�ec�A�*

loss�/=gU��       �	�De�ec�A�*

loss|"�=�!�       �	/f�ec�A�*

lossi�5=+lr       �	f�f�ec�A�*

loss��c=/���       �	��g�ec�A�*

loss7�=��D�       �	��h�ec�A�*

loss�#�=�A]       �	��i�ec�A�*

loss,��=�X       �	��j�ec�A�*

loss��=ȏ�       �	w�k�ec�A�*

losso��=DMn3       �	r4l�ec�A�*

loss��S<R�B�       �	$Cm�ec�A�*

losso�X<����       �	�m�ec�A�*

loss��b=�J/X       �	��n�ec�A�*

loss�8�=�}�p       �	�ho�ec�A�*

loss \�=a޴       �	tp�ec�A�*

loss]=�g��       �	
q�ec�A�*

loss l�=��!       �	T�q�ec�A�*

loss�8=�ǟ�       �	�Ar�ec�A�*

loss1�[=g;�       �	
�r�ec�A�*

loss��=�GM'       �	�ns�ec�A�*

lossoՁ=G%:       �	�t�ec�A�*

loss��=7��       �	ƥt�ec�A�*

loss�=�~k�       �	��u�ec�A�*

loss��<���       �	�mv�ec�A�*

loss��<l�o       �	1w�ec�A�*

loss��F=�f�'       �	ӡw�ec�A�*

loss�g�=@E7�       �	�Ex�ec�A�*

loss@�<5�"?       �	��x�ec�A�*

loss���<�md�       �	Loy�ec�A�*

losshD>e"       �	�z�ec�A�*

loss]��<G,��       �	��z�ec�A�*

loss��=�&N"       �	2>{�ec�A�*

loss*�<���s       �	��{�ec�A�*

loss�R�=�pA�       �	�|�ec�A�*

loss� �=H�L       �	W%}�ec�A�*

loss�>[=ciF       �	�}�ec�A�*

loss�W>I��       �	uW~�ec�A�*

loss��=l�s       �	�3�ec�A�*

lossTf�=:�N	       �	��ec�A�*

loss�t2=��H       �	�r��ec�A�*

lossʪ�=�<G�       �	���ec�A�*

loss�|�=��       �	����ec�A�*

lossi��=��L|       �	"7��ec�A�*

lossϻ�;Z.��       �	Oɂ�ec�A�*

loss�e=�e�       �	4i��ec�A�*

loss�=a9�G       �	����ec�A�*

loss�N=�^��       �	����ec�A�*

loss�o	<���       �	�8��ec�A�*

loss�t"=��p       �	�х�ec�A�*

loss��<�>3}       �	�i��ec�A�*

loss���=�^�       �	�7��ec�A�*

lossJ��=�`�h       �	�݇�ec�A�*

loss)�%=��       �	���ec�A�*

loss�=Ƒx       �	y?��ec�A�*

lossxm�=��m�       �	�ډ�ec�A�*

loss��8=OC�       �	�z��ec�A�*

loss?��=<7#�       �	b��ec�A�*

loss��=||��       �	V��ec�A�*

loss/(=����       �	 ���ec�A�*

lossx�L<=�=t       �	�R��ec�A�*

loss�¹<2�y&       �	���ec�A�*

loss1=�       �	P���ec�A�*

loss(��=)�ir       �	$��ec�A�*

lossV��<xCR�       �	Ը��ec�A�*

loss��<T��       �	T��ec�A�*

loss�3l<�N�]       �	P���ec�A�*

losssy<C�       �	����ec�A�*

loss���=       �	�6��ec�A�*

loss�E>B#8�       �	ڒ�ec�A�*

loss�_�=d��       �	�r��ec�A�*

loss/Q>裯�       �	'��ec�A�*

loss�M�<�r       �	`���ec�A�*

loss=��`�       �	}A��ec�A�*

loss*>I���       �	���ec�A�*

loss��P=�8}n       �	����ec�A�*

loss��0=Ϡ�O       �	c^��ec�A�*

loss%�T=��
       �	���ec�A�*

loss�۳=�_U�       �	����ec�A�*

loss�'�<���~       �	O��ec�A�*

loss65=�[�       �	��ec�A�*

loss#>���       �	����ec�A�*

lossZ:>��r       �	M/��ec�A�*

lossP�=��       �	uʛ�ec�A�*

lossb�	=�vg�       �	io��ec�A�*

loss���<t[nC       �	]��ec�A�*

lossIuB=A�+�       �	����ec�A�*

loss��=�V�v       �	轞�ec�A�*

loss~�<�ʐ       �	�f��ec�A�*

loss��<�Yf�       �	U��ec�A�*

lossq<�w��       �	غ��ec�A�*

loss^Š=�{W:       �	�\��ec�A�*

lossı�=���@       �	.���ec�A�*

loss�ʗ<>�|       �	����ec�A�*

loss#��=j[�       �	*R��ec�A�*

loss�X�<R�?       �	����ec�A�*

loss�X�=���       �	����ec�A�*

loss� �<��9�       �	�C��ec�A�*

loss{�=����       �	d��ec�A�*

loss�%�<0ɺ~       �	���ec�A�*

loss�l=MZ��       �	�ç�ec�A�*

loss���=8���       �	�r��ec�A�*

loss� o=��:�       �	���ec�A�*

loss�p�<��Nx       �	�ש�ec�A�*

loss��@<��/       �	4���ec�A�*

loss-z=�+�       �	R���ec�A�*

loss��=���X       �	M��ec�A�*

loss6"�=d|�       �	����ec�A�*

lossA>Vr�        �	x���ec�A�*

loss��">�0η       �	�F��ec�A�*

loss�ʛ=u��H       �	���ec�A�*

loss���=g� �       �	����ec�A�*

lossE��=�       �	a3��ec�A�*

loss��<�?       �	�ް�ec�A�*

loss7�0=�=�       �	Ǆ��ec�A�*

loss�c=VY&       �	H��ec�A�*

loss�	=��[�       �	X��ec�A�*

loss��X=A��B       �	A���ec�A�*

loss�oS=ҭ�       �	YP��ec�A�*

loss�F�=�;3�       �	`��ec�A�*

loss�=�;#�\�       �	k~��ec�A�*

loss�y=��n�       �	���ec�A�*

loss�E�=�'�       �	����ec�A�*

loss��<f��5       �	�E��ec�A�*

lossm��=����       �	�ٷ�ec�A�*

loss��<�L�       �	Kr��ec�A�*

loss�o�=��3       �	B��ec�A�*

loss�~.>�[��       �	���ec�A�*

loss��=����       �	j���ec�A�*

loss�><��	�       �	�1��ec�A�*

loss��=`D�       �	vݻ�ec�A�*

loss�=����       �	f���ec�A�*

loss3�,=��       �	�Q��ec�A�*

loss�Q=�g.�       �	����ec�A�*

loss"\=N��       �	����ec�A�*

lossM<=�'G       �	Z��ec�A�*

lossvT�=/[��       �	���ec�A�*

loss�&�<��ޢ       �	Uh��ec�A�*

loss�Խ<����       �	���ec�A�*

loss�٨=~W/%       �	����ec�A�*

lossA�<Sc�       �	'l��ec�A�*

loss� =�,~�       �	Q��ec�A�*

loss��.<2�^�       �	����ec�A�*

loss@�=ԩdF       �	�_��ec�A�*

lossx�<�=c       �	���ec�A�*

loss�ֆ<{!��       �	����ec�A�*

loss���;q��l       �	�E��ec�A�*

loss��T=�i�       �	X���ec�A�*

lossr��<�(&-       �	����ec�A�*

loss�x<�C*       �	2��ec�A�*

loss_[V=���       �	����ec�A�*

lossm�P=џ�       �	�S��ec�A�*

loss�$b=��       �	%���ec�A�*

loss�:�<��i       �	���ec�A�*

lossυ�=����       �	���ec�A�*

loss�ee=��6Q       �	����ec�A�*

loss|��=�n��       �	�U��ec�A�*

lossfv�<�Q       �	s���ec�A�*

loss|\1=��i�       �	����ec�A�*

loss;[�=6iP       �	�(��ec�A�*

loss�e<�n��       �	~���ec�A�*

lossJ��<���       �	9_��ec�A�*

loss�^<u�o)       �	1	��ec�A�*

loss�HD<ēS9       �	����ec�A�*

loss��;H22j       �	�H��ec�A�*

lossN�=�\       �	O���ec�A�*

loss(��=xiԆ       �	����ec�A�*

loss|��=���       �	���ec�A�*

loss�G�<�	       �	y���ec�A�*

loss�;��d       �	{K��ec�A�*

loss�)U:!ˍY       �	&��ec�A�*

loss:Z�<5��2       �	����ec�A�*

lossl|�=��M       �	�j��ec�A�*

loss�L=3��       �	��ec�A�*

loss��;�z8:       �	ϻ��ec�A�*

loss��=�U�       �	'i��ec�A�*

loss�8�>��H�       �	,D��ec�A�*

loss׹�<�m��       �	c���ec�A�*

loss�='       �	����ec�A�*

loss��=:#�-       �	px��ec�A�	*

loss;>M2m^       �	d��ec�A�	*

loss�C�<�3��       �	����ec�A�	*

loss��<����       �	�d��ec�A�	*

loss.�>��{�       �	 ��ec�A�	*

loss=�4=���       �	���ec�A�	*

loss�V=�<\       �	�F��ec�A�	*

loss���=�l�W       �	d���ec�A�	*

lossx(�<>m5       �	e���ec�A�	*

loss���=��w�       �	�1��ec�A�	*

loss��=g���       �	O���ec�A�	*

loss/S]=�g�4       �	in��ec�A�	*

loss�Q�=C�G�       �	;��ec�A�	*

loss��=$?+       �	����ec�A�	*

lossIn�=Q_1�       �	�p��ec�A�	*

loss��=p5jW       �	Z��ec�A�	*

loss��>({��       �	c��ec�A�	*

loss��H=�h�       �	����ec�A�	*

loss�6�<��}�       �	U���ec�A�	*

lossJŜ=���|       �	 ���ec�A�	*

loss�'=ot&       �	Y5��ec�A�	*

loss�e<��       �	9���ec�A�	*

lossnx`<ք;�       �	n��ec�A�	*

lossû�<�YL8       �	���ec�A�	*

loss�W�<���       �	|���ec�A�	*

loss�k=�8Y�       �	�1��ec�A�	*

loss�Q>�w9�       �	}���ec�A�	*

loss��=��=�       �	�m��ec�A�	*

loss��
=�-       �	���ec�A�	*

loss%}=�JK�       �	7���ec�A�	*

loss��Y=߼�;       �	�J��ec�A�	*

lossIgA<�_��       �	����ec�A�	*

loss��s=�V�       �	���ec�A�	*

losssQ�<���a       �	�1��ec�A�	*

lossTC�<8`a:       �	|���ec�A�	*

lossS�V= ��       �	Yi��ec�A�	*

lossz>�f�_       �	�	��ec�A�	*

loss�� =��s       �	[���ec�A�	*

loss��=7v!5       �	����ec�A�	*

lossZ��=#bK       �	$��ec�A�	*

loss��C=�+��       �	0���ec�A�	*

loss��>=��j       �	`V��ec�A�	*

loss�}=����       �	����ec�A�	*

lossp��=�K�2       �	����ec�A�	*

lossx��=K�eW       �	K>��ec�A�	*

lossV7<Kŭ9       �	����ec�A�	*

loss�|=�:}�       �	<���ec�A�	*

loss��<s�       �	D��ec�A�	*

loss��G=/V�.       �	����ec�A�	*

lossOBa=]�G*       �	��ec�A�	*

loss�z�=�ƞ       �	��ec�A�	*

loss�� >z�xn       �	(C�ec�A�	*

loss��=�夘       �	���ec�A�	*

loss�J`=ak��       �	sJ�ec�A�	*

lossm��<�4        �	V��ec�A�	*

loss%3z=�S��       �	���ec�A�	*

losslӶ<���       �	�0�ec�A�	*

lossFG�=�'Bq       �	���ec�A�	*

loss,�=���       �	
��ec�A�	*

loss�W=�9L�       �	(�ec�A�	*

loss��L=G�T       �	���ec�A�	*

loss�,�=�� �       �	�g�ec�A�	*

loss4�*=��o       �	��ec�A�	*

loss~�<�j��       �	|��ec�A�	*

loss���=S��       �	;4�ec�A�	*

lossM�;K�Vj       �	h��ec�A�	*

loss�"=� ��       �	d�ec�A�	*

loss.�7=��]       �	&��ec�A�	*

loss�x=�>u       �	��ec�A�	*

loss�D�=EWO       �	�/ �ec�A�	*

loss�K>*-("       �	� �ec�A�	*

loss73<��9       �	]!�ec�A�	*

lossѩ�=���       �	��!�ec�A�	*

loss��=S�]       �	!�"�ec�A�	*

loss��<ŵ��       �	>$#�ec�A�	*

loss;��<K �       �	�#�ec�A�	*

loss��=�f��       �	�L$�ec�A�	*

loss��=#s�       �	~�$�ec�A�	*

lossʆ>z�Y�       �	l�%�ec�A�	*

loss��E<0��o       �	��&�ec�A�	*

loss]�<��Z       �	�'�ec�A�	*

lossÑ�=�Ȼ�       �	�-(�ec�A�	*

loss���=A7�       �	I�(�ec�A�	*

loss��a=t&�h       �	��)�ec�A�	*

loss�H�=վ�7       �	
-*�ec�A�	*

loss��
=��O�       �	��*�ec�A�	*

loss�)=�CQ�       �	�_+�ec�A�	*

loss@޴=����       �	�;,�ec�A�	*

lossق�=�`U9       �	J�,�ec�A�	*

loss�f�=�}�L       �	fk-�ec�A�	*

loss��:=s��n       �	�.�ec�A�	*

loss�7<3	�\       �	+�.�ec�A�	*

loss�='���       �	5B/�ec�A�	*

lossp�=2%/�       �	�/�ec�A�	*

loss��l=�i��       �	Ɗ0�ec�A�	*

loss�� >Xk��       �	�-1�ec�A�	*

lossܩ�<��iN       �	��1�ec�A�	*

lossF�=����       �	΋2�ec�A�	*

lossL��;1�H       �	�-3�ec�A�	*

loss�Te<�DH�       �	F�3�ec�A�	*

loss�<#=Ϡ��       �	2s4�ec�A�	*

loss�3�<ڜ1       �	�5�ec�A�	*

loss��J>�h�       �	#h6�ec�A�	*

loss�`�=���       �	_7�ec�A�	*

loss6ޕ;M��R       �	�7�ec�A�	*

lossu�	=)}�       �	�Z8�ec�A�	*

loss�P�<��a       �	��8�ec�A�	*

lossp�<x� �       �	¥9�ec�A�	*

loss;�=Ψi�       �	MM:�ec�A�	*

loss���=⣎�       �	��:�ec�A�	*

loss}��<*}qi       �	T�;�ec�A�	*

loss壺;��       �	.<�ec�A�	*

lossE`�<�/Hm       �	�<�ec�A�	*

loss�<�kQ       �	�j=�ec�A�	*

loss�$+<�Z_�       �	�>�ec�A�	*

loss��-=Wq��       �	��>�ec�A�	*

loss�]x<9��       �	VH?�ec�A�	*

loss��=.��       �	��?�ec�A�	*

loss���=�<��       �	��@�ec�A�	*

loss���<�@'       �	`>A�ec�A�	*

loss��=��<�       �	j�A�ec�A�	*

loss�C�<zZ�q       �	�rB�ec�A�	*

lossC�U<e1�       �	C�ec�A�	*

lossDC�<
�x�       �	�C�ec�A�	*

loss�ca<��L       �	�GD�ec�A�	*

loss�P>���       �	%$E�ec�A�	*

loss`�d=��       �	#�E�ec�A�	*

loss�q�<�q �       �	�XF�ec�A�	*

loss&Մ=��       �	��F�ec�A�
*

loss���=S9�       �	��G�ec�A�
*

losss9�=��vZ       �	�#H�ec�A�
*

lossm�<��73       �	ӼH�ec�A�
*

lossƖ;���       �	&TI�ec�A�
*

loss<�=�T�       �	��I�ec�A�
*

loss��=�Ȱ       �	:�J�ec�A�
*

loss&Nw<���b       �	&5K�ec�A�
*

loss�qd=�8c1       �	��K�ec�A�
*

loss�~<b{       �	wL�ec�A�
*

loss&1=�I�       �	^M�ec�A�
*

loss��<}h>�       �	 �M�ec�A�
*

loss)|�<�	wP       �	p`N�ec�A�
*

loss�f-=��_�       �	�O�ec�A�
*

loss�_�<�P��       �	��O�ec�A�
*

loss\��=�k�       �	$FP�ec�A�
*

lossS��=I۟d       �	��P�ec�A�
*

loss��=�jI       �	q�Q�ec�A�
*

loss���;&���       �	�7R�ec�A�
*

loss	�>��       �	��R�ec�A�
*

loss�E�<�;�       �	��S�ec�A�
*

loss���<����       �	�.T�ec�A�
*

loss��<)�S�       �	�T�ec�A�
*

lossM==J�       �	KwU�ec�A�
*

loss��=����       �	%#V�ec�A�
*

loss ��<����       �	-�V�ec�A�
*

lossCc>ŋa�       �	lW�ec�A�
*

loss�T<E��L       �	X�ec�A�
*

lossf=�S�       �	9�X�ec�A�
*

loss���=k�$�       �	ZY�ec�A�
*

loss��<"/��       �	�vZ�ec�A�
*

loss���=f���       �	�[�ec�A�
*

loss�g =�ܘ�       �	=�[�ec�A�
*

loss_Ԯ=�0�6       �	�f\�ec�A�
*

loss�*<���       �	�+]�ec�A�
*

loss�w�<PT\       �	�]�ec�A�
*

loss׶�<�ϙ       �	c_^�ec�A�
*

loss�Y�<��b"       �	_�ec�A�
*

loss�S�=@aC�       �	��_�ec�A�
*

loss
G=�I�W       �	g`�ec�A�
*

losswW�<���       �	�a�ec�A�
*

loss��^=Q�F       �	��a�ec�A�
*

loss��z=��G       �	�8b�ec�A�
*

loss�s=߅$>       �	�b�ec�A�
*

loss��{=)Dl&       �	Oyc�ec�A�
*

loss���<;���       �	d�ec�A�
*

lossn �<�v��       �	8�d�ec�A�
*

loss���<+YAU       �	W_e�ec�A�
*

lossOr�<�>�       �	f�ec�A�
*

loss�k�=��ʁ       �	��f�ec�A�
*

loss��=��w�       �	CYg�ec�A�
*

loss�̖<Fy��       �	ah�ec�A�
*

lossK9=Ƽ�}       �	DNi�ec�A�
*

loss�׃=��7�       �	�@j�ec�A�
*

lossoζ=����       �	��j�ec�A�
*

loss_К=��       �	k�k�ec�A�
*

loss{�j<F>�       �	��l�ec�A�
*

loss8 =#�       �	�rm�ec�A�
*

loss�- <�q��       �	Yn�ec�A�
*

lossE<5*�6       �	�n�ec�A�
*

loss=sC<Q�2�       �	zRo�ec�A�
*

loss��F;h�d       �	!�o�ec�A�
*

loss,S�=1M~       �	��p�ec�A�
*

loss�4�=k���       �	�q�ec�A�
*

lossm&/>���       �	8�q�ec�A�
*

loss�*>ю       �	1\r�ec�A�
*

loss�[�<U�wn       �	� s�ec�A�
*

lossa��=��       �	�s�ec�A�
*

loss.A�;�-N$       �	�At�ec�A�
*

lossJ<.��       �	+�t�ec�A�
*

lossz�/<(K!       �	�uu�ec�A�
*

loss��P<]q35       �	$v�ec�A�
*

loss]�>=SO�?       �	 �v�ec�A�
*

loss���<�
��       �	NCw�ec�A�
*

losslÚ=���       �	�w�ec�A�
*

loss�>�=,��O       �	)vx�ec�A�
*

loss���<�Z��       �	�Ky�ec�A�
*

loss�Y=��O       �	�Dz�ec�A�
*

loss�dA=i���       �	M�z�ec�A�
*

loss?�=��I�       �	��{�ec�A�
*

loss�7=���       �	�|�ec�A�
*

loss�,�=ME�=       �	�|�ec�A�
*

loss-8P=�U�7       �	zr}�ec�A�
*

loss)�p=��Pc       �	�8~�ec�A�
*

lossq�J<�#U�       �	Y�~�ec�A�
*

losst�='�(�       �	Ox�ec�A�
*

loss�f=n�ʥ       �	��ec�A�
*

loss�\�;@��       �	ͫ��ec�A�
*

loss��<M��       �	�j��ec�A�
*

lossN�<���'       �	���ec�A�
*

loss�= �X       �	*���ec�A�
*

loss��=���Q       �	�A��ec�A�
*

lossF��;&��t       �	8؃�ec�A�
*

loss(0�<��8       �	i��ec�A�
*

lossSr'>���       �	���ec�A�
*

lossK��;JG�       �	B���ec�A�
*

loss{�X=�^c&       �	�N��ec�A�
*

lossM=�<լ�+       �	���ec�A�
*

lossqg�=���!       �	Eև�ec�A�
*

loss�ݽ<�07�       �	
i��ec�A�
*

loss��%<�Ƣo       �	g��ec�A�
*

loss�P}<P�       �	֬��ec�A�
*

lossF�=�       �	�K��ec�A�
*

loss���=	p��       �	���ec�A�
*

loss}?$<8�H       �	⏋�ec�A�
*

lossCH<��3       �	k)��ec�A�
*

loss{<�+�       �	~Ȍ�ec�A�
*

loss���<�c�P       �	#g��ec�A�
*

loss��U=���       �		��ec�A�
*

loss��=^�       �	����ec�A�
*

loss��=�0��       �	�I��ec�A�
*

loss��=�=       �	�ߏ�ec�A�
*

loss��;=^a��       �	;5��ec�A�
*

loss�"�<�R�D       �	�ˑ�ec�A�
*

loss��;<��8�       �	1|��ec�A�
*

losso5;Qy       �	��ec�A�
*

loss�uL=8�Ԅ       �	����ec�A�
*

loss�D&=��       �	�I��ec�A�
*

loss�IM=����       �	�۔�ec�A�
*

loss 7O=g��       �	qv��ec�A�
*

loss7�<���       �	��ec�A�
*

loss=��<�O�       �	����ec�A�
*

loss{&^<�~�"       �	�S��ec�A�
*

loss3�	<*���       �	���ec�A�
*

loss�=#wN;       �	4���ec�A�
*

loss�lm;���       �	'��ec�A�
*

loss=��	       �	���ec�A�
*

loss�8<=�w��       �	�T��ec�A�
*

loss娀=��t        �	���ec�A�*

loss��#=��_/       �	����ec�A�*

lossxl�<�^9"       �	����ec�A�*

loss�*�=��       �	�+��ec�A�*

loss*��<��       �	�ĝ�ec�A�*

loss�ڦ<�ϸ       �	\��ec�A�*

loss��<Fk�       �	k��ec�A�*

loss>=n��}       �	犟�ec�A�*

loss�[�=A�X�       �	�#��ec�A�*

lossCK<>A
�       �	�ޠ�ec�A�*

loss�B�=	> �       �	i��ec�A�*

lossȹ(==��Z       �	�w��ec�A�*

lossw��<��,       �	�$��ec�A�*

loss3��<���1       �	����ec�A�*

loss��;kSS	       �	}X��ec�A�*

loss���<6mŇ       �	���ec�A�*

loss&�?==�N        �	\���ec�A�*

lossF)";��E       �	G;��ec�A�*

lossܥ�<���p       �	K��ec�A�*

loss�M=t^=�       �	���ec�A�*

loss���=���       �	X���ec�A�*

lossw��<���\       �	����ec�A�*

loss�L�<gO�       �	A���ec�A�*

loss2�<K�s�       �	["��ec�A�*

loss!�;��:       �	]ë�ec�A�*

loss^=��       �	B��ec�A�*

loss
:�<�`�       �	���ec�A�*

losso =�I�       �	O��ec�A�*

loss�6D<C��        �	����ec�A�*

loss��<Y���       �	����ec�A�*

lossQ�=��       �	�H��ec�A�*

loss�3j=�m	       �	_��ec�A�*

loss�B<o
��       �	���ec�A�*

loss��<Q�~       �	0ײ�ec�A�*

loss<�x=y|m�       �	�l��ec�A�*

lossɟr=�h�       �	y��ec�A�*

loss� �=*p�       �	u��ec�A�*

loss�z>=1�	       �	dY��ec�A�*

loss37:<-�<�       �	�Q��ec�A�*

loss�@<
��       �	(��ec�A�*

loss:	u=jg�       �	�F��ec�A�*

loss��>1��       �	$���ec�A�*

loss�&<R�        �	�p��ec�A�*

loss��5=����       �	��ec�A�*

loss4<=mo��       �	N��ec�A�*

loss "�=
�.-       �	����ec�A�*

lossf�I<���%       �	4��ec�A�*

loss���;����       �	�Ծ�ec�A�*

loss�zL<ԃ$Q       �	&p��ec�A�*

loss5=�:N@       �	���ec�A�*

loss=��<[�)�       �	����ec�A�*

loss�,0=�H9       �	y���ec�A�*

loss,�%=Q⤏       �	���ec�A�*

loss6��<�f       �	g'��ec�A�*

loss��<͎       �	����ec�A�*

loss��?=�}��       �	Ae��ec�A�*

loss��<�̘�       �	���ec�A�*

lossj��=<FҢ       �	5���ec�A�*

loss��.=��u@       �	1?��ec�A�*

lossԧ!=[�       �	Q���ec�A�*

loss$ޚ=�c0�       �	v��ec�A�*

loss�$�=�U��       �	��ec�A�*

lossME�=�8R�       �	Y���ec�A�*

loss�<�w�       �	�>��ec�A�*

losslN|=(���       �	����ec�A�*

loss�ɇ=�ǹ;       �	�s��ec�A�*

lossqP=���|       �	���ec�A�*

loss؄u=+F�1       �	����ec�A�*

lossqo~=q�$�       �	QK��ec�A�*

loss�(b=�(       �	 ���ec�A�*

loss�d�:ч��       �	���ec�A�*

loss.�d=^*��       �	'��ec�A�*

loss;�p=��`       �	:���ec�A�*

loss���;[�*       �	d��ec�A�*

lossvp�<�,g�       �	���ec�A�*

loss���;GVe�       �	ծ��ec�A�*

loss��4<ߖ%�       �	�E��ec�A�*

loss/��=!��       �	����ec�A�*

losso��=P���       �	����ec�A�*

loss��;=���       �	�$��ec�A�*

lossG==A,%       �	���ec�A�*

loss���<F���       �	CT��ec�A�*

loss��;�~��       �	����ec�A�*

loss�]L<���>       �	]���ec�A�*

loss�..=���       �	���ec�A�*

loss%��;p^       �	{���ec�A�*

loss��	<��V       �	�Q��ec�A�*

loss ��;_m       �	B��ec�A�*

loss�E�=L�[       �	c���ec�A�*

loss;X�<��
        �	+i��ec�A�*

loss�>'�'<       �	��ec�A�*

lossA2�=o�:�       �	����ec�A�*

loss�J=���       �	;S��ec�A�*

loss<�q<�a��       �	���ec�A�*

lossX��=�9�2       �	:���ec�A�*

loss�	Y;ټ)2       �	-?��ec�A�*

loss�~�=���       �	����ec�A�*

losso��<�y@�       �	�}��ec�A�*

loss�Ng>���       �	m��ec�A�*

loss�;F8P^       �	���ec�A�*

loss+�=5�U       �	�G��ec�A�*

lossa�=?�       �	M���ec�A�*

loss^��=��rI       �	mp��ec�A�*

loss��z<�&       �		��ec�A�*

lossaŸ<V|v�       �	w���ec�A�*

loss�%�<�/��       �	�I��ec�A�*

loss�FI<c��       �	����ec�A�*

lossN,e<t�N       �	ޓ��ec�A�*

lossVZ=�e$        �	�1��ec�A�*

loss���<�x`c       �	
���ec�A�*

loss��T<�G�       �	"q��ec�A�*

loss��B=���       �	��ec�A�*

lossJ�=� ˊ       �	M���ec�A�*

loss�0)=�oʊ       �	_y��ec�A�*

loss35�=�e       �	���ec�A�*

lossnx�<�l�       �	����ec�A�*

lossF�g<��8       �	�P��ec�A�*

loss�o�<}��       �	"���ec�A�*

lossp�=фG       �	���ec�A�*

lossL.o=�;ë       �	*9��ec�A�*

loss�iy=����       �	����ec�A�*

loss�y�=����       �	����ec�A�*

loss�i�=ں%       �	���ec�A�*

lossÁ�<��~       �	Ǻ��ec�A�*

loss�A�<�|�o       �	}X��ec�A�*

loss��;x�       �	����ec�A�*

lossz�k<�!�       �	���ec�A�*

loss!"=Ga�>       �	�)��ec�A�*

loss��<��'5       �	����ec�A�*

loss\��<	&�       �	�\��ec�A�*

loss]
�<s-��       �	����ec�A�*

loss&j2=A�b�       �	����ec�A�*

loss��w;+s[�       �	�8��ec�A�*

loss�A�<�W�       �	:���ec�A�*

loss?�=��g       �	d��ec�A�*

loss�<�Rz�       �	 ��ec�A�*

lossqa=[h�       �	R���ec�A�*

loss׌q=�[�       �	�:��ec�A�*

loss4�=���k       �	R���ec�A�*

lossx�<���       �	�n��ec�A�*

lossź�=���M       �	h��ec�A�*

loss���=҈X       �	���ec�A�*

loss�ٯ=�2�       �	jP��ec�A�*

loss)ݓ<E�<       �	����ec�A�*

loss`��<�5�       �	I���ec�A�*

loss)�=4�       �	D��ec�A�*

loss�(K=��3       �	����ec�A�*

loss�X=�m��       �	����ec�A�*

loss�N�;�JF�       �	���ec�A�*

loss_8�<Va{Z       �	����ec�A�*

loss���;���v       �	�w �ec�A�*

loss�f>�\       �	�J�ec�A�*

loss(�=���I       �	���ec�A�*

loss�R�<��       �	�w�ec�A�*

lossѺ�<�/�<       �	��ec�A�*

lossS�<�>o       �	��ec�A�*

lossh�=R�       �	`X�ec�A�*

lossѥ=O�k�       �	��ec�A�*

loss��
=-�G       �	Χ�ec�A�*

loss� �=7�       �	�L�ec�A�*

loss �u=ƭ��       �	��ec�A�*

loss�<S��       �	��ec�A�*

loss_8�<V!d�       �	F"�ec�A�*

lossl0=���W       �	���ec�A�*

lossx��<��m�       �	�^	�ec�A�*

loss��z=�hב       �	��	�ec�A�*

loss/]:=���       �	p�
�ec�A�*

loss��<l���       �	�4�ec�A�*

loss���<�%q       �	-��ec�A�*

loss�Q5=�ӹX       �	ʩ�ec�A�*

loss�<�<>p��       �	�W�ec�A�*

loss/�^<U�6�       �	g��ec�A�*

loss*O�<֦�A       �	���ec�A�*

loss���<^�$�       �	�!�ec�A�*

loss���=�1$g       �	��ec�A�*

loss)�,>�{[�       �	I��ec�A�*

loss|�7<QChC       �	t%�ec�A�*

lossV��;����       �	���ec�A�*

loss-E?=��G�       �	[a�ec�A�*

loss�ȑ<Q��       �	���ec�A�*

loss���=��0       �	:��ec�A�*

loss-ŋ=/M�g       �	�)�ec�A�*

loss��<��_=       �	���ec�A�*

loss��]=b�CX       �	sh�ec�A�*

loss��=URb       �	e��ec�A�*

loss�!5<"�       �	s��ec�A�*

loss��=B9�       �	;6�ec�A�*

loss6�g=���z       �	���ec�A�*

loss��d=�z       �	l��ec�A�*

loss�Ζ<�c�#       �	>>�ec�A�*

lossU,=���       �	h �ec�A�*

lossK��<�m��       �	{��ec�A�*

loss���=��.       �	�Q�ec�A�*

lossWH�<�E       �	���ec�A�*

loss6�=
���       �	'��ec�A�*

losst#=/#M       �	�t�ec�A�*

loss�Po=� V       �	��ec�A�*

loss`�m=A�ԯ       �	H��ec�A�*

loss��(=^�       �	�b�ec�A�*

loss�9�=%��       �	e��ec�A�*

loss���=xw��       �	�� �ec�A�*

loss6�3<�.��       �	
K!�ec�A�*

loss�͒=P#^       �	4�!�ec�A�*

loss�=A���       �	G�"�ec�A�*

lossh��=$��q       �	�{#�ec�A�*

loss7�<7Xɘ       �	F%$�ec�A�*

loss� >�'�	       �	X�$�ec�A�*

loss�ַ=y�~�       �	
f%�ec�A�*

loss|�f=��cG       �	�&�ec�A�*

lossS)T<��E^       �	
M'�ec�A�*

loss�<�\T�       �	��'�ec�A�*

loss&�:=���r       �	��(�ec�A�*

loss�l=��_       �	��)�ec�A�*

loss�EL=�߯       �	�v*�ec�A�*

loss1�=�{
�       �	�+�ec�A�*

loss��<���       �	z�,�ec�A�*

loss*g�=�l(�       �	^-�ec�A�*

loss�>�L�Z       �	!.�ec�A�*

loss0��=w�k�       �	�/�ec�A�*

lossO�V=ie�/       �	�T0�ec�A�*

loss�6"=�p_�       �	�<1�ec�A�*

loss��o=�0&       �	�1�ec�A�*

loss�"=�K        �	>�2�ec�A�*

loss*-�<�\j�       �	;�3�ec�A�*

loss��7=�lv]       �	�i4�ec�A�*

loss�'�<� Q	       �	X5�ec�A�*

lossE�]=D�*+       �	��5�ec�A�*

loss��<����       �	'36�ec�A�*

loss��<�m��       �	��6�ec�A�*

loss��5<���       �	�p7�ec�A�*

loss��}<`O��       �		8�ec�A�*

loss�T=�}�%       �	��8�ec�A�*

loss=��<]�"V       �	�?9�ec�A�*

loss=h�=�]�       �	b�9�ec�A�*

lossv�Z=н��       �	x:�ec�A�*

lossn��=6XZ6       �	�;�ec�A�*

loss�d>
x�A       �	��;�ec�A�*

loss��<���Z       �	�I<�ec�A�*

lossd|"=��Xd       �	B�<�ec�A�*

loss�B(>��8�       �	��=�ec�A�*

lossQq=۟�       �	�>�ec�A�*

loss�x=��+       �	�v?�ec�A�*

loss��G=��O%       �	�@�ec�A�*

loss�o�=%e�#       �	_�@�ec�A�*

loss
�4<Iu�       �	`XA�ec�A�*

loss�V�<��%�       �	#�A�ec�A�*

loss誊<3J��       �	K�B�ec�A�*

lossz9�=6H�S       �	�1C�ec�A�*

loss��
=2^|7       �	��C�ec�A�*

loss.ƕ=���        �	?rD�ec�A�*

loss!X�=Y��       �	�E�ec�A�*

loss(Q =@�po       �	��E�ec�A�*

loss��=Ŕ��       �	�JF�ec�A�*

loss��<+f/�       �	��F�ec�A�*

loss��=��(�       �	w�G�ec�A�*

lossDt@<b��       �	�H�ec�A�*

lossH#�=����       �	÷H�ec�A�*

loss:=�<Y�a       �	�NI�ec�A�*

loss�*�=�H�\       �	J�I�ec�A�*

lossO�J=��)       �	.�J�ec�A�*

loss`�=��       �	Z.K�ec�A�*

lossva=
��       �	T�K�ec�A�*

loss��<hQp       �	�L�ec�A�*

loss%��=��Z�       �	�M�ec�A�*

lossd�<U̲�       �	��M�ec�A�*

lossW�_=9�       �	�NN�ec�A�*

loss�=��NB       �	��N�ec�A�*

loss���<�)       �	~�O�ec�A�*

loss�k<��۸       �	�.P�ec�A�*

loss���=%�.       �	>�P�ec�A�*

loss%��<�E�       �	erQ�ec�A�*

loss��=���F       �	VR�ec�A�*

loss�S>�<�       �	��R�ec�A�*

loss-�P=�×�       �	inS�ec�A�*

loss��,=~��       �	�T�ec�A�*

loss���<i��T       �	�T�ec�A�*

loss��=��       �	�fU�ec�A�*

loss�h�<�M       �	V�ec�A�*

lossN:=����       �	{W�ec�A�*

lossƛ< F�       �	��W�ec�A�*

loss?T5=ʞ�       �	�gX�ec�A�*

loss'1=B��       �	UY�ec�A�*

loss��=��߿       �	��Y�ec�A�*

loss��s=�pN       �	fZ�ec�A�*

lossv�=�c��       �	�[�ec�A�*

loss#Ռ<�H       �	��[�ec�A�*

loss�f	=
Pe>       �	�v\�ec�A�*

loss%�=}1�       �	]�ec�A�*

loss��b=iS�       �	��]�ec�A�*

loss�<C�@�       �	\r^�ec�A�*

lossn�h<yY�S       �	*_�ec�A�*

loss�3D<�3b#       �	#�_�ec�A�*

loss��8=8�/�       �	_`�ec�A�*

loss\]�<\��       �	��`�ec�A�*

loss��;LS��       �	�a�ec�A�*

loss�i<"7H       �	i;b�ec�A�*

loss���<�C2       �	��b�ec�A�*

loss�x=nb
�       �	c�c�ec�A�*

lossà�<�Hۉ       �	\�d�ec�A�*

loss�=�a�       �	�/e�ec�A�*

loss���<L>D       �	o�e�ec�A�*

loss�>="��       �	��f�ec�A�*

loss�Н=�3�       �	�+g�ec�A�*

lossk�=�7o�       �	6�g�ec�A�*

loss4��=FG�       �	5zh�ec�A�*

lossnAk;Q��A       �	 &i�ec�A�*

lossx��=dz��       �	tDj�ec�A�*

lossnp=�=�W       �	�j�ec�A�*

loss=��=M��       �	$�k�ec�A�*

loss���=d���       �	2>l�ec�A�*

lossY6!=�>0�       �	�l�ec�A�*

loss��=�o$       �	؟m�ec�A�*

lossӄ=��       �	Gn�ec�A�*

loss�]<�[Ɂ       �	�n�ec�A�*

loss��i=t�q2       �	!�o�ec�A�*

loss��J:�^�       �	�>p�ec�A�*

loss�r<�x       �	��p�ec�A�*

lossO�<&��!       �	��q�ec�A�*

loss��U=��       �	�5r�ec�A�*

loss\I�;�T
N       �	��r�ec�A�*

loss��<�=�n       �	'�s�ec�A�*

lossJb=�2Z       �	�'t�ec�A�*

lossM.q;���       �	i�t�ec�A�*

loss��[<�R+�       �	qu�ec�A�*

lossV�<�g]�       �	�1v�ec�A�*

lossj�;�u�       �	��v�ec�A�*

loss�u <�<       �	�rw�ec�A�*

lossQ�A<��(b       �	�x�ec�A�*

lossy<�/�       �	X�x�ec�A�*

loss�(x;�y�o       �	My�ec�A�*

lossm��;�9       �	��y�ec�A�*

loss�4Z99�w|       �	.�z�ec�A�*

loss)�<O��       �	�2{�ec�A�*

loss�p=3�[       �	g�{�ec�A�*

lossE"9��       �	fi|�ec�A�*

losssl_9�xL       �	�	}�ec�A�*

loss�v<s5�       �	�}�ec�A�*

loss�='�2"       �	R`~�ec�A�*

loss��<�z�,       �	��ec�A�*

loss�
;UL       �	"��ec�A�*

loss`5<qs�/       �	�T��ec�A�*

loss��P>u�8=       �	L���ec�A�*

loss��;(�       �	����ec�A�*

loss�ǽ<Д��       �	@��ec�A�*

lossHC=��9�       �	!��ec�A�*

loss�c=ZNs7       �	Z~��ec�A�*

lossw�<o%�-       �	� ��ec�A�*

loss�߫<a-��       �	3�ec�A�*

loss��,>j�M�       �	�w��ec�A�*

loss�=v�       �	���ec�A�*

loss�)=ꄎ�       �	�ʆ�ec�A�*

loss�͘<�v�       �	w��ec�A�*

loss	5=\��       �	���ec�A�*

loss�0�=��/�       �	Lǈ�ec�A�*

loss�cv=*��Z       �	j��ec�A�*

loss�f�=�î       �	��ec�A�*

loss{Z�=
��V       �	W���ec�A�*

loss��=�PӐ       �	�S��ec�A�*

loss�=�`�       �	����ec�A�*

loss>Y=c��       �	Y���ec�A�*

loss�<�=L;)       �	�I��ec�A�*

lossʄ<��{#       �	���ec�A�*

loss҂�<����       �	����ec�A�*

loss��=J.Y�       �	%A��ec�A�*

loss�^>=�n*       �	��ec�A�*

loss�/�;��       �	ٕ��ec�A�*

loss_?=,��       �	@��ec�A�*

lossD3�<Xm�       �	��ec�A�*

loss=�w+�       �	����ec�A�*

loss`�c=�z�-       �	�>��ec�A�*

loss,��={�G       �	���ec�A�*

loss`=?ٹv       �	����ec�A�*

loss���<���       �	�0��ec�A�*

lossLE;=RW�       �	�̕�ec�A�*

loss�l=��	�       �	�v��ec�A�*

lossa�J<��x       �	a��ec�A�*

lossi�=Bg
       �	�x��ec�A�*

loss�%V<��       �	+��ec�A�*

loss)<���&       �	�ϙ�ec�A�*

loss.�=KKe�       �		o��ec�A�*

loss
�=+�!�       �	n��ec�A�*

loss�̌=F���       �	����ec�A�*

loss3�;��)�       �	�R��ec�A�*

loss2�;<ES -       �	���ec�A�*

lossa�=���       �	ō��ec�A�*

loss1��=>��       �	�-��ec�A�*

lossa�=�W?       �	BϞ�ec�A�*

loss�=b�2       �	�m��ec�A�*

loss=�=� �       �	;��ec�A�*

loss���<M�U�       �	c���ec�A�*

loss�a�<D_       �	�V��ec�A�*

loss/Y.<�b�       �	���ec�A�*

lossi2�<�YH�       �	�ec�A�*

loss}��<x�@�       �	+K��ec�A�*

loss!�<�3.�       �	8���ec�A�*

loss42=b�T�       �	�:��ec�A�*

lossm�@=bL u       �	����ec�A�*

losseY=߳\R       �	?t��ec�A�*

losst[c<�i>Q       �	M��ec�A�*

loss�9�<��       �	!���ec�A�*

lossm�=3��       �	�V��ec�A�*

loss���<!�=�       �	7���ec�A�*

loss��B<%Q"�       �	���ec�A�*

loss,Ea;,^T�       �	_@��ec�A�*

lossv�=��,#       �	'���ec�A�*

loss�<"P�       �	����ec�A�*

loss�=�}�u       �	�-��ec�A�*

lossP�=��o       �	K���ec�A�*

loss�='�       �	#e��ec�A�*

loss��<����       �	���ec�A�*

loss��^<���I       �	$���ec�A�*

losse�=��@�       �	�4��ec�A�*

loss��=#a       �	����ec�A�*

loss�6�=f���       �	�w��ec�A�*

loss��>�Bt       �	<��ec�A�*

lossz#9=�0ϲ       �	����ec�A�*

loss��=��8�       �	�R��ec�A�*

loss�s�<�ҋv       �	H���ec�A�*

loss���;C�mo       �	����ec�A�*

loss��=���       �	79��ec�A�*

loss��S<��#�       �	����ec�A�*

loss��O=5V-       �	����ec�A�*

lossH��<y�       �	{.��ec�A�*

loss�]0=��       �	����ec�A�*

loss�u=�J&       �		q��ec�A�*

loss��=� �       �	���ec�A�*

lossI�=���y       �	U���ec�A�*

loss�%�<���       �	�=��ec�A�*

lossW�=5�)�       �	����ec�A�*

loss�T�;O���       �	 r��ec�A�*

loss��I=���       �	��ec�A�*

loss�=̴o�       �	����ec�A�*

lossS)�<ŝY�       �	�:��ec�A�*

loss�h�=n��       �	,���ec�A�*

lossx��=K�z�       �	ao��ec�A�*

loss��O;���       �	���ec�A�*

loss��<>��u       �	���ec�A�*

loss=�2=�tr2       �	B��ec�A�*

loss��[<�XR�       �	����ec�A�*

loss� =�H�       �	j���ec�A�*

loss�m8=3"�t       �	>��ec�A�*

lossԹ�<D���       �	E���ec�A�*

loss4�7<�,�       �	���ec�A�*

lossɍ<��@�       �	���ec�A�*

loss�A�=WH       �	����ec�A�*

loss�S�;%�       �	����ec�A�*

lossOK >���       �	J%��ec�A�*

lossp��=���a       �	;���ec�A�*

loss>ԅ;�!�       �	�e��ec�A�*

lossa��;�K}       �	����ec�A�*

loss���;u{��       �	���ec�A�*

loss�0�=v�0       �	@��ec�A�*

loss	2=x:u�       �	r���ec�A�*

lossh�:=�$�       �	:A��ec�A�*

loss�{�<e:       �	���ec�A�*

loss���=>� �       �	�/��ec�A�*

lossx0=�*B!       �	����ec�A�*

lossR�8=<�L       �	E*��ec�A�*

loss���<�lY�       �	���ec�A�*

losst��=<!�       �	�w��ec�A�*

loss�F=�;��       �	����ec�A�*

loss�a=�s0�       �	�8��ec�A�*

loss���<S�u�       �	���ec�A�*

loss�xE=�ܱ9       �	?s��ec�A�*

loss�:)=9�c�       �	���ec�A�*

loss=�<��       �	q���ec�A�*

lossZN<��       �	r���ec�A�*

loss�A>=�D�       �	Cq��ec�A�*

loss}6b<h!g       �	���ec�A�*

loss�+<�� �       �	2��ec�A�*

lossV,D<"М       �	����ec�A�*

loss!=s�*       �	�s��ec�A�*

lossx.=�> �       �	0��ec�A�*

loss�ħ;��e       �	����ec�A�*

loss�@�=��Oo       �	'K��ec�A�*

loss��=����       �	7���ec�A�*

loss��D=��^       �	����ec�A�*

loss`0�<���       �	Q3��ec�A�*

loss*.<&�       �	L���ec�A�*

loss��3=�cvE       �	_��ec�A�*

loss0 "==[�'       �	 �ec�A�*

loss�xV<���       �	�� �ec�A�*

loss.�7<��Ǚ       �	vS�ec�A�*

loss��=6���       �	���ec�A�*

loss��\=����       �	���ec�A�*

lossa��<RB�8       �	4��ec�A�*

lossJN=«       �	Xr�ec�A�*

lossRAj<Q�       �	��ec�A�*

loss�<��A�       �	��ec�A�*

lossz=��       �	�5�ec�A�*

loss���=lޔ       �	X��ec�A�*

loss��=c��#       �	�]�ec�A�*

lossC�;&��       �	���ec�A�*

loss�`�<��W�       �	8��ec�A�*

loss̠<�0�	       �	�	�ec�A�*

loss��v<F�{�       �	}�	�ec�A�*

loss���;�~��       �	�U
�ec�A�*

lossJ�<Ħ��       �	5�
�ec�A�*

loss\��<���       �	H��ec�A�*

lossX=<k�#�       �	�!�ec�A�*

loss�0�<�Kz^       �	���ec�A�*

loss8�=��B       �	c^�ec�A�*

loss��<Uk�       �	��ec�A�*

loss=��=��E       �	F��ec�A�*

lossՃ�<�)�       �	�V�ec�A�*

loss�A=s��U       �	��ec�A�*

loss��:�t-       �	z��ec�A�*

loss��;|_��       �	S%�ec�A�*

loss��y:�7�       �	d��ec�A�*

loss�]o=(Zl       �	Qh�ec�A�*

loss=`3��       �	��ec�A�*

loss�z=�       �	K��ec�A�*

loss'�<�R�       �	H�ec�A�*

loss��<�#�       �	/��ec�A�*

loss!��=���       �	�{�ec�A�*

lossjW�<�ُ�       �	0�ec�A�*

loss\j=�	e       �	���ec�A�*

lossf�;0��       �	�B�ec�A�*

loss�<��Q       �	7��ec�A�*

loss*��=�ri�       �	�|�ec�A�*

loss?8�<�~��       �	/�ec�A�*

loss�j=�`D�       �	��ec�A�*

loss��=��B�       �	�D�ec�A�*

lossh+=m4��       �	
��ec�A�*

loss���<���O       �	�n�ec�A�*

lossLH0=�I�       �	��ec�A�*

loss��<k���       �	f��ec�A�*

loss$�<m���       �	�G�ec�A�*

loss�J�;N3�       �	�\�ec�A�*

loss��h=l��k       �	���ec�A�*

loss���<��$       �	j��ec�A�*

loss�1�;
��       �	 �ec�A�*

loss��w;�1�B       �	� �ec�A�*

loss*�;F4a       �	XU!�ec�A�*

lossƬ�<���-       �	��!�ec�A�*

loss�#=^(�       �	H�"�ec�A�*

loss"+=�x��       �	��#�ec�A�*

loss\>�?f:       �	~�$�ec�A�*

loss29<�{�       �	�1%�ec�A�*

lossu�<(��       �	��%�ec�A�*

loss."�;6J�$       �	�z&�ec�A�*

loss	�-<��R�       �	6Y'�ec�A�*

loss��{;���       �	��'�ec�A�*

loss�/�;H��       �	��(�ec�A�*

loss��K=�u�O       �	�)�ec�A�*

loss�u�<7E�       �	�)�ec�A�*

loss
A�=�l��       �	�m*�ec�A�*

lossf�m=VQfT       �	-+�ec�A�*

loss��<*��       �	K�+�ec�A�*

lossM �<�Rv       �	8I,�ec�A�*

loss&��=�%%�       �	B�,�ec�A�*

lossnY=�à�       �	��-�ec�A�*

lossH�<��       �	E*.�ec�A�*

loss�{�=���       �	��.�ec�A�*

lossx�<iJ�1       �	�|/�ec�A�*

loss�d=��2       �	!0�ec�A�*

loss6��<�Yw�       �	:y1�ec�A�*

lossc��<ő9�       �	\2�ec�A�*

lossܒ><[�̚       �	��2�ec�A�*

loss���;;.e?       �	�Y3�ec�A�*

loss��=u��       �	��3�ec�A�*

loss�3�<�B!�       �	�4�ec�A�*

loss�/=\ ��       �	5�ec�A�*

loss4ح<�B�       �	��5�ec�A�*

loss�"J;�L<       �	=G6�ec�A�*

loss�<e��       �	n�6�ec�A�*

lossv��<�:�       �	�x7�ec�A�*

loss��;,��&       �	8�ec�A�*

loss�O<湇       �	��8�ec�A�*

loss��z= g�Z       �	6;9�ec�A�*

loss��=ߘ^       �	l�9�ec�A�*

loss���=�p�       �	�d:�ec�A�*

loss��;�a       �	;�ec�A�*

loss�C�:�g       �	��;�ec�A�*

loss	_�=�4Q       �	�5<�ec�A�*

loss��K<���3       �	��<�ec�A�*

loss���;��?�       �	�w=�ec�A�*

lossH>�;����       �	�>�ec�A�*

loss
;j��-       �	2�>�ec�A�*

loss[H=Zi+       �	�P?�ec�A�*

lossl	= �WD       �	;�?�ec�A�*

lossn�<Y�#       �	�x@�ec�A�*

loss�0�<�Wf�       �	�[A�ec�A�*

loss���=���$       �	��A�ec�A�*

lossW� =G'O#       �	lC�ec�A�*

loss��><^�@       �	qD�ec�A�*

loss���<�Hj       �	�D�ec�A�*

loss��;N���       �	�JE�ec�A�*

loss��w=�8 �       �	��E�ec�A�*

losscI =l�Ē       �	X�F�ec�A�*

loss�>\�,�       �	�4G�ec�A�*

loss�c>39       �	��G�ec�A�*

loss��K<$#t       �	zmH�ec�A�*

lossT_�=sk8�       �	�I�ec�A�*

loss/EL=���       �	�I�ec�A�*

loss9%==�bB       �	fjJ�ec�A�*

loss�.�<��       �	@K�ec�A�*

loss�tg<����       �	��K�ec�A�*

loss��<O\��       �	�YL�ec�A�*

lossR��<It�       �	0�L�ec�A�*

loss�E=�ЁK       �	3�M�ec�A�*

losszB�=?�I�       �	� N�ec�A�*

loss*0=o^Pl       �	7�N�ec�A�*

loss�֬<���       �	�gO�ec�A�*

loss:�9=��:�       �	q�O�ec�A�*

loss;ޚ<S���       �	G�P�ec�A�*

loss��<�I5       �	V(Q�ec�A�*

loss�,�<I�I       �	��Q�ec�A�*

lossdl�=�$��       �	 VR�ec�A�*

loss�ݔ=�Mz�       �	c�R�ec�A�*

loss��=6Qc�       �	��S�ec�A�*

lossMJM=��       �	�$T�ec�A�*

loss�
>�ub5       �	2�T�ec�A�*

loss��<��l       �	 ]U�ec�A�*

lossRؐ<��X       �	(�U�ec�A�*

lossn��;���       �	��V�ec�A�*

loss��a=vHn�       �	�3W�ec�A�*

loss�/9<���       �	��W�ec�A�*

loss��=�»       �	^hX�ec�A�*

loss�э=_t��       �	tY�ec�A�*

lossC==�ɉ       �	��Y�ec�A�*

loss��?=�I$       �	�JZ�ec�A�*

loss��%=i�Ci       �	%�Z�ec�A�*

loss��=Q�e�       �	��[�ec�A�*

loss�|<��	       �	F�\�ec�A�*

loss���<�A��       �	4L]�ec�A�*

loss ǌ<�W�       �	��]�ec�A�*

lossΈ%=�=��       �	��^�ec�A�*

loss�(|<��$S       �	�7_�ec�A�*

loss�t=�o��       �	��_�ec�A�*

loss �5=V��       �	��`�ec�A�*

loss�=���0       �	$Da�ec�A�*

loss�\`;}�!       �	��a�ec�A�*

loss��/<�i�       �	C�b�ec�A�*

loss�Ӄ=�ħU       �	�.c�ec�A�*

loss<>=@�       �	2�c�ec�A�*

lossd-/=`�U�       �	�jd�ec�A�*

lossS�b=2��G       �	�e�ec�A�*

loss$�F=��       �	<�e�ec�A�*

lossHJh<�c�       �	�Zf�ec�A�*

loss���;����       �	�g�ec�A�*

lossS�<�W��       �	��g�ec�A�*

loss�{<V�p�       �	�Sh�ec�A�*

loss��<=d�|�       �	a�h�ec�A�*

lossf��=_       �	�i�ec�A�*

loss=}Z=8�f�       �	�=j�ec�A�*

loss��g;&|K�       �	��j�ec�A�*

loss�;���       �	��k�ec�A�*

loss�?�<\�o�       �	�+l�ec�A�*

loss�N�<R�R�       �	G�l�ec�A�*

loss��+<��S�       �	�vm�ec�A�*

loss��s<�>q�       �	�n�ec�A�*

lossX��<d�r       �	��n�ec�A�*

loss}��<���       �	�o�ec�A�*

loss}6�;���       �	o+p�ec�A�*

loss%X<_>�5       �	/�p�ec�A�*

loss�W3;B e       �	�Zq�ec�A�*

loss�%�=�j,       �	��q�ec�A�*

loss�|�;<��x       �	.�r�ec�A�*

loss?�=C�:�       �	��t�ec�A�*

loss
'�;q�5�       �	B�u�ec�A�*

loss|f�=��.       �	�Fv�ec�A�*

loss��<Y]��       �	�Rw�ec�A�*

lossQB=5��       �	!x�ec�A�*

loss!\�<�8       �	�x�ec�A�*

lossl�]<�.       �	__y�ec�A�*

loss��.<��,       �	*�y�ec�A�*

loss��=�J��       �	~�z�ec�A�*

loss�<�=�Ʋ�       �	F\{�ec�A�*

loss2��<��/\       �	�|�ec�A�*

loss��;_ᾊ       �	
�|�ec�A�*

loss�b�=ھ܈       �	'f}�ec�A�*

lossv�<l�h       �	%~�ec�A�*

loss��<���       �	��~�ec�A�*

lossf�G;^UX       �	���ec�A�*

losst�0<<U\�       �	���ec�A�*

lossm,�=1�       �	� ��ec�A�*

loss\��<�7�       �	"Ɓ�ec�A�*

loss$r%=f�       �	ۆ��ec�A�*

lossC�=���       �	M,��ec�A�*

loss��#<I��7       �	�Ճ�ec�A�*

loss���<#�2t       �	w���ec�A�*

loss��;(6�#       �	~R��ec�A�*

loss��:�a�       �	��ec�A�*

loss��;=��       �	����ec�A�*

loss�H(;���       �	cB��ec�A�*

loss8��;�P��       �	Qڇ�ec�A�*

loss�k<��cf       �	�t��ec�A�*

loss�Mu=%���       �	�N��ec�A�*

loss�};=M���       �	���ec�A�*

loss���<�?�D       �	���ec�A�*

loss(L�=���f       �	~��ec�A�*

lossf�j=�UF       �	-���ec�A�*

loss ��<��9�       �	/O��ec�A�*

lossl1A<j���       �	B��ec�A�*

lossњ<ӂs�       �	ӆ��ec�A�*

lossψ=���       �	$��ec�A�*

loss��@<�AY       �	����ec�A�*

loss���=kG<�       �	�U��ec�A�*

loss��<]?�       �	L���ec�A�*

lossa��=�]R�       �	���ec�A�*

loss�jE=��       �	 7��ec�A�*

loss;�<�"�       �	ؑ�ec�A�*

loss#t<5$��       �	&p��ec�A�*

lossW��<��|h       �	���ec�A�*

loss�W�<��3�       �	���ec�A�*

loss���;���       �	]7��ec�A�*

loss�y�<'3��       �	����ec�A�*

loss�vl=ZWǞ       �	&���ec�A�*

loss��;�� 8       �	�"��ec�A�*

loss��<.�qv       �	����ec�A�*

loss�<�^F       �	T��ec�A�*

lossWlH=SL       �	!��ec�A�*

loss��<���E       �	Ҍ��ec�A�*

loss�h�=���       �	X��ec�A�*

loss�ڳ<�HЀ       �	����ec�A�*

loss(C�<�c��       �	�_��ec�A�*

loss��<@��+       �	/���ec�A�*

lossذ�=��U       �	O���ec�A�*

loss���;`Q��       �	�>��ec�A�*

loss��=�F�       �	;���ec�A�*

loss�"+=$ol}       �	m��ec�A�*

loss���<s���       �	��ec�A�*

loss�Y�<�tT       �	j.��ec�A�*

lossH�O=k@Ҟ       �	�̟�ec�A�*

loss�?�;)5F�       �	=|��ec�A�*

lossI�<
)߽       �	�&��ec�A�*

loss�X=���       �	2ʡ�ec�A�*

lossq�=_�y       �	�s��ec�A�*

loss�@�<E-��       �	���ec�A�*

loss�C�;��L`       �	ۿ��ec�A�*

loss���<}l3Z       �	`��ec�A�*

loss�;O��       �	���ec�A�*

loss�hc;��ڮ       �	���ec�A�*

loss��W=gi��       �	�=��ec�A�*

loss�C=�T�_       �	�}��ec�A�*

loss,@<{��       �	>!��ec�A�*

loss�]�=���G       �	kԨ�ec�A�*

loss�<j�       �	큩�ec�A�*

lossc0�<;�,       �	�)��ec�A�*

loss�=5=��'>       �	#ת�ec�A�*

loss�'�<��H�       �	t{��ec�A�*

lossq=*W��       �	"��ec�A�*

loss��<8V[V       �	E���ec�A�*

losso\�<�u�       �	�l��ec�A�*

lossqs>�8       �	?���ec�A�*

lossTb�<ٶ��       �	B���ec�A�*

loss��=Z�b�       �	oI��ec�A�*

loss��G<�h       �	W��ec�A�*

lossx�6<��w       �	����ec�A�*

losst�J<��       �	�s��ec�A�*

loss�&�;'Q       �	���ec�A�*

lossH�	=iaΛ       �	:W��ec�A�*

loss�Ҍ=`��/       �	���ec�A�*

loss`[V<R��>       �	u���ec�A�*

loss�V�<*>|�       �	�,��ec�A�*

loss�O�<��6�       �	}˶�ec�A�*

loss�5=��O       �	bg��ec�A�*

loss$�=0Τ       �	;��ec�A�*

loss�a-=��
�       �	ط��ec�A�*

lossP6�=�!|�       �	�R��ec�A�*

loss)V�<1!P�       �	w��ec�A�*

loss��
={�E       �	3���ec�A�*

lossЧ>|�D�       �	�+��ec�A�*

lossM��<'TI       �	ʻ�ec�A�*

loss1�f;8+��       �	�^��ec�A�*

loss��<�Xv       �	���ec�A�*

loss$=a��       �	����ec�A�*

loss:��=))�^       �	�!��ec�A�*

lossv~=f@       �	o���ec�A�*

loss_�<��       �	�U��ec�A�*

loss
�=��e       �	���ec�A�*

loss��=S�       �	���ec�A�*

lossWHL<��R�       �	{/��ec�A�*

loss̤l=�z�       �	����ec�A�*

loss�f�=͈       �	���ec�A�*

loss՗�<�`�       �	�!��ec�A�*

loss���:u��j       �	2���ec�A�*

loss/�L=��       �	�{��ec�A�*

loss �<��8       �	�)��ec�A�*

loss���=|]��       �	����ec�A�*

loss�;�
dH       �	W^��ec�A�*

lossX7=�|W�       �	����ec�A�*

loss$V<^&W�       �	¢��ec�A�*

loss�
r=�7��       �	=��ec�A�*

loss�h�<pVB�       �	j���ec�A�*

loss�^�<nP

       �	�z��ec�A�*

loss(t=+��       �	���ec�A�*

loss��=��RS       �	���ec�A�*

loss�Tz:����       �	���ec�A�*

lossIpM;�G�       �	���ec�A�*

lossŝq<�p$O       �	����ec�A�*

loss�N�<�E�L       �	�U��ec�A�*

loss�=��A       �	H���ec�A�*

loss�u<<��:4       �	C���ec�A�*

loss�p�=~�j       �	�e��ec�A�*

losspi<��[       �	r���ec�A�*

loss��/=�p\�       �	}���ec�A�*

loss�=(�a�       �	ZF��ec�A�*

lossr4b=|rH       �	����ec�A�*

losslQM<Ž�       �	~��ec�A�*

loss���<D�&>       �	���ec�A�*

lossVp	<���       �	���ec�A�*

loss��<�U_
       �	�L��ec�A�*

loss6�=A���       �	����ec�A�*

loss�<�<?��[       �	؂��ec�A�*

loss ��<̐KT       �	q��ec�A�*

loss�=�;J}       �	ú��ec�A�*

loss���=.�D2       �	����ec�A�*

loss-7};��_f       �	J��ec�A�*

loss��;�c?�       �	����ec�A�*

loss��<	?~       �	���ec�A�*

loss���=d^�       �	�3��ec�A�*

loss<�
�5       �	����ec�A�*

loss��<ob�       �	I���ec�A�*

loss��3<?�       �	�'��ec�A�*

lossFd�=�y�       �	����ec�A�*

lossS)%=$`�       �	�n��ec�A�*

loss�|=grz�       �	S	��ec�A�*

loss�><�>�4       �	���ec�A�*

loss�O�<d�\�       �	M��ec�A�*

loss��0<�qO       �	����ec�A�*

lossf/e={_       �	���ec�A�*

lossŐo=	.��       �	�(��ec�A�*

loss!L�<Mx��       �	����ec�A�*

lossJL�;"3v�       �	�n��ec�A�*

loss�>p<�A�       �	��ec�A�*

loss,a=u3,�       �	֫��ec�A�*

lossMf�<�h�       �	�D��ec�A�*

loss 7I<�ot�       �	����ec�A�*

loss�;�?}       �	:u��ec�A�*

loss��:Ux�       �	���ec�A�*

loss1{=5��       �	����ec�A�*

loss�37=�l��       �	\r��ec�A�*

loss
 q=��x�       �	c	��ec�A�*

loss7M�= d�	       �	8���ec�A�*

loss��9=�\�       �	&9��ec�A�*

loss1�;�Y�       �	����ec�A�*

loss
֚<	�r       �	�}��ec�A�*

loss�}B=�BS       �	w��ec�A�*

loss}9=�RN5       �	K���ec�A�*

lossQl�<p�#       �	k��ec�A�*

lossTnJ=�B5u       �	O��ec�A�*

lossF>�=�v       �	����ec�A�*

loss� �;�8��       �	����ec�A�*

loss��<���f       �	���ec�A�*

lossi��<R>�E       �	���ec�A�*

loss�[>tO�       �	�6��ec�A�*

loss�Vj;���q       �	����ec�A�*

loss��?="�rv       �	����ec�A�*

loss�}^<�ap?       �	���ec�A�*

loss	$�<C�so       �	+1��ec�A�*

loss��<�R�       �	����ec�A�*

loss�x�=�@�       �	Q���ec�A�*

loss��c<�V�       �	)$��ec�A�*

loss薎;^<�       �	���ec�A�*

loss���=~�S       �	�g��ec�A�*

loss�t=t|�P       �	���ec�A�*

loss<�<�:q5       �	L���ec�A�*

loss�6=�`�Z       �	�D��ec�A�*

loss�K*<H��I       �	����ec�A�*

loss�;�;ʥf       �	υ��ec�A�*

loss�ɤ;n�';       �	���ec�A�*

loss���=n�       �	E���ec�A�*

loss��<�f0�       �	~R��ec�A�*

lossS�I=��p)       �	����ec�A�*

loss��<��@r       �	����ec�A�*

losshT,;��       �	R)��ec�A�*

lossa�I<��$       �	����ec�A�*

loss*�n=�L�~       �	=_��ec�A�*

loss[O=��
=       �	{���ec�A�*

loss�U�<�`�o       �	%� �ec�A�*

loss�=�?[�       �	)�ec�A�*

loss�	>_�'       �	���ec�A�*

loss��C=���       �	Ed�ec�A�*

loss��=��U       �		�ec�A�*

loss��<�0��       �	&��ec�A�*

loss�!h=m �i       �	'��ec�A�*

loss��;�O+�       �	�D�ec�A�*

loss<t#;Ϧ�       �	���ec�A�*

lossT� =u��       �	yv�ec�A�*

loss71�:��9       �	+�ec�A�*

lossZ]1=�J       �	M��ec�A�*

loss�ۂ=>�?       �	�Y�ec�A�*

loss��<�>�`       �	;��ec�A�*

lossI�;�e       �	��	�ec�A�*

loss���<М       �	�8
�ec�A�*

lossOZB;J�^       �	�
�ec�A�*

lossvx�=s�T       �	dx�ec�A�*

loss7/U<��*R       �	��ec�A�*

loss�$<ڸ W       �	���ec�A�*

loss�#�<:�;
       �	M�ec�A�*

loss���<b,��       �	���ec�A�*

loss�2 =-��       �	˅�ec�A�*

lossn��;{j�       �	6#�ec�A�*

lossv��<w��       �	���ec�A�*

lossyt�<h��       �	*W�ec�A�*

lossx��<��D       �	���ec�A�*

loss���<��       �	���ec�A�*

lossCN=k�ǳ       �	�+�ec�A�*

loss!ؔ;�l��       �	���ec�A�*

loss��<�,Nh       �	,a�ec�A�*

loss��
=����       �	���ec�A�*

lossC��;ʤ�I       �	Ƥ�ec�A�*

loss��<���M       �	A�ec�A�*

loss��j;�[�4       �	��ec�A�*

loss'r=�$�       �	2u�ec�A�*

loss��;�-�       �	�ec�A�*

loss
8�=����       �	Ͱ�ec�A�*

lossz�W<�%��       �	@L�ec�A�*

lossC�B<���1       �	O��ec�A�*

loss��0<���E       �	��ec�A�*

loss�G�<-��       �	#�ec�A�*

loss��`;�|*z       �	���ec�A�*

loss_�<}jT       �	�`�ec�A�*

loss�;��5       �	~�ec�A�*

loss|g<;v���       �	e��ec�A�*

lossAl�<�D?�       �	�H�ec�A�*

loss�#<��8�       �	���ec�A�*

lossm��;6�2M       �	%��ec�A�*

loss�$<cշ�       �	�5�ec�A�*

loss�1>2��       �	���ec�A�*

loss�#�=ܲo�       �	� �ec�A�*

loss��:)(�       �	U3!�ec�A�*

loss��)=_�۾       �	��!�ec�A�*

loss�P�:�^"�       �	��"�ec�A�*

loss<�+=�S��       �	�.#�ec�A�*

loss���;���       �	c�#�ec�A�*

loss) ;�?w�       �	%u$�ec�A�*

lossɰ�;�z�P       �	�%�ec�A�*

loss<a�;r�
       �	M�%�ec�A�*

loss��;G���       �	�e&�ec�A�*

loss,G<5���       �	�'�ec�A�*

loss�K=K�       �	z�'�ec�A�*

loss�^�:uz~6       �	G(�ec�A�*

lossԺ�:�p^�       �	��(�ec�A�*

loss��X=ćNV       �	g�)�ec�A�*

lossr�#=�%C�       �	(F*�ec�A�*

loss_6�;?0V       �	"�*�ec�A�*

loss zH9'�O       �	��+�ec�A�*

loss�q>=5�Z�       �	_),�ec�A�*

lossd�
>6؈w       �	k�,�ec�A�*

lossn��;�yK6       �	]p-�ec�A�*

lossa)�<�`J�       �	
.�ec�A�*

loss���=����       �	7�.�ec�A�*

loss&��<�n��       �	�D/�ec�A�*

loss���<5
�h       �	��/�ec�A�*

loss�Q�;Ze�O       �	$0�ec�A�*

loss��f=�       �	v1�ec�A�*

lossߡ=
P       �	$�1�ec�A�*

losst�t<��tt       �	�R2�ec�A�*

loss!]�<��=       �	��2�ec�A�*

loss̄B<HW"�       �	�3�ec�A�*

lossf�;=h�       �	�4�ec�A�*

losso��=��jL       �	�4�ec�A�*

loss���<�m       �	�V5�ec�A�*

loss7�=DÅ       �	W�5�ec�A�*

lossF=�<�X�!       �	T�6�ec�A�*

loss�=z���       �	v57�ec�A�*

loss��7=�P�^       �	��7�ec�A�*

lossf$C=��w       �	]k8�ec�A�*

loss�^=�Z%D       �	7�8�ec�A�*

loss[�o<��zi       �	�9�ec�A�*

lossF}�<����       �	�B:�ec�A�*

lossjz�<����       �	[�:�ec�A�*

loss<}n;4޴�       �	��;�ec�A�*

lossiY<$��       �	#1<�ec�A�*

loss.=T<}-!       �	��<�ec�A�*

loss�]=l�	�       �	�r=�ec�A�*

loss1��;A�\       �	�>�ec�A�*

loss9.�<�       �	M�>�ec�A�*

loss�`=h|       �	�R?�ec�A�*

loss}^<ߘ0       �	c�?�ec�A�*

loss���<��h�       �	R�@�ec�A�*

loss�2;�9,�       �	9A�ec�A�*

lossC�j;>��       �	3�A�ec�A�*

loss��D<v}�       �	��B�ec�A�*

loss=�p;���       �	
-C�ec�A�*

loss��I<���       �	��C�ec�A�*

loss��=���       �	�uD�ec�A�*

lossҨ�=3R5       �	�E�ec�A�*

lossR1=r�_�       �	)�E�ec�A�*

loss�<0jx�       �	GrF�ec�A�*

loss,'=�\�2       �	�G�ec�A�*

loss�X=�Ӎi       �	�G�ec�A�*

loss�=|!D�       �	kbH�ec�A�*

loss�"<�h�       �	�I�ec�A�*

lossw�=��_&       �	�I�ec�A�*

loss�T�=�Ox9       �	�FJ�ec�A�*

lossj�N<z�|3       �	 �J�ec�A�*

loss32
=���       �	t�K�ec�A�*

lossO��;���,       �	9L�ec�A�*

loss�>�<DZ��       �	*�L�ec�A�*

loss\��<@�2       �	�c�ec�A�*

loss2a =���(       �	U�c�ec�A�*

loss=5=�D��       �	C7d�ec�A�*

loss��<G�O       �	�d�ec�A�*

loss��<Rs�       �	�he�ec�A�*

loss�w=��g       �	\�e�ec�A�*

loss�9�<w�,       �	�f�ec�A�*

loss²�<��*       �	�1g�ec�A�*

loss~ǐ=[Ѫ�       �	 �g�ec�A�*

loss�l�<��B�       �	�eh�ec�A�*

loss��F;9�l*       �	U�i�ec�A�*

loss<�=�ǩ�       �	Kuj�ec�A�*

loss,`�<��ƍ       �	$
k�ec�A�*

loss㬟<����       �	D�k�ec�A�*

loss�hG<�P�       �	�Zl�ec�A�*

loss���<`�͖       �	6m�ec�A�*

loss�o;Z�>]       �	��m�ec�A�*

losst+<���       �	��n�ec�A�*

loss��<u���       �	�Uo�ec�A�*

loss�1=ʒu       �	��o�ec�A�*

loss��p=��y�       �	�p�ec�A�*

loss�=����       �	�[q�ec�A�*

lossh�:i;�       �	�r�ec�A�*

loss��[=�x5�       �	��r�ec�A�*

loss��	=��2       �	]ks�ec�A�*

loss�1.=&�A       �	K�t�ec�A�*

lossH��;��O�       �	D�u�ec�A�*

loss�M�<�G�       �	ov�ec�A�*

loss��<�K2�       �	�_x�ec�A�*

loss��:"��6       �	nRy�ec�A�*

loss�<�uhv       �	�
z�ec�A�*

lossa�;���M       �	j�z�ec�A�*

loss�l�;no�C       �	C<{�ec�A�*

lossϺ=�/       �	��{�ec�A�*

loss2T<�i�       �	�x|�ec�A�*

loss�Y~<���j       �	�}�ec�A�*

loss`<ׄ�g       �	�}�ec�A�*

lossN�<&���       �	/M~�ec�A�*

loss,��=��0Y       �	��~�ec�A�*

loss;�=���y       �	���ec�A�*

loss1�<�0       �	���ec�A�*

loss��=l�O       �	>���ec�A�*

loss#��;�%�"       �	'.��ec�A�*

lossU�<"�~x       �	�ƃ�ec�A�*

loss�u<�&�       �	�_��ec�A�*

loss�Z�<�5o�       �	����ec�A�*

loss$�(=�	V�       �	���ec�A�*

loss6�<�5�{       �	�1��ec�A�*

loss('<��5-       �	�׆�ec�A�*

loss}�:��y2       �	�o��ec�A�*

loss���;���       �	�	��ec�A�*

loss\f�<���       �	먈�ec�A�*

loss3S�:�ϖ]       �	�D��ec�A�*

loss|2=l��*       �	���ec�A�*

lossr5(;�=       �	���ec�A�*

loss��}:j���       �	c%��ec�A�*

loss�:Yv       �	�΋�ec�A�*

loss��=�"b       �	�u��ec�A�*

lossx�<���       �	���ec�A�*

lossC3=W'�       �	M���ec�A�*

loss
�<�Z��       �	%\��ec�A�*

loss�3<���       �	����ec�A�*

loss�=���6       �	$���ec�A�*

loss��<9i�D       �	�7��ec�A�*

loss3b;
��8       �	�Ԑ�ec�A�*

loss�F�9N>��       �	�x��ec�A�*

loss���<!-F       �	���ec�A�*

loss�Z<��i&       �	0���ec�A�*

lossݨ�<x��       �	�Q��ec�A�*

lossV��<�r�       �	���ec�A�*

loss���;v��X       �	J~��ec�A�*

loss�'�<i�Z       �	���ec�A�*

loss��N=ߒ��       �	u���ec�A�*

loss���=�(σ       �	�F��ec�A�*

loss�f8<�>?0       �	�ݖ�ec�A�*

loss-��<���       �	�t��ec�A�*

lossa�a;���q       �	b��ec�A�*

lossaV�;���       �	㦘�ec�A�*

loss�cW<tu��       �	�>��ec�A�*

loss�+<�F�0       �	ՙ�ec�A�*

loss�e�;�AL�       �	�v��ec�A�*

loss}�<V8��       �	E��ec�A�*

loss#�<���       �	]���ec�A�*

loss��/:�od@       �	0H��ec�A�*

loss=�;��̠       �	����ec�A�*

lossW�V=g͌�       �	_y��ec�A�*

loss���<҈�       �	k��ec�A�*

loss�02<!a/�       �	Ѯ��ec�A�*

lossJt�<�?Jn       �	H��ec�A�*

loss�<����       �	���ec�A�*

loss!��<�XR�       �	k���ec�A�*

loss/��=!�7       �	���ec�A�*

loss@:�;]v�'       �	г��ec�A�*

loss|�=9��        �	�L��ec�A�*

loss�J�;�0       �	d��ec�A�*

lossE6�;�.Ϊ       �	V���ec�A�*

loss�i�<z���       �	z��ec�A�*

loss)h�;��j       �	帤�ec�A�*

lossA�=�2L�       �	HP��ec�A�*

loss�u�;��       �	T��ec�A�*

loss���<ͺ��       �	�|��ec�A�*

loss�4I:'�d�       �	b��ec�A�*

loss�[Y;���       �	"���ec�A�*

loss)�!<Ő�4       �	E��ec�A�*

loss88�;���E       �	Lݨ�ec�A�*

loss�n�<�b       �	�t��ec�A�*

loss�:�<��Y       �	���ec�A�*

lossSln<:���       �	ڬ��ec�A�*

loss��)=q,H       �	�F��ec�A�*

loss$�N<��A       �	���ec�A�*

loss�Ɋ<S��       �	�|��ec�A�*

lossXu;���       �	���ec�A�*

lossd-=��L       �	�ܭ�ec�A�*

losseO�;��	       �	�|��ec�A�*

loss$�;8:֖       �	���ec�A�*

lossIs<=�D�i       �	���ec�A�*

loss���;���       �	kG��ec�A�*

loss<fb=�޽-       �	#ܰ�ec�A�*

loss�t�<>"Y�       �	�q��ec�A�*

loss#��;4�+       �	��ec�A�*

losss3�<��t�       �	ݲ��ec�A�*

loss�!B=�M�       �	jO��ec�A�*

loss =
�       �	l��ec�A�*

loss��P=���       �	�~��ec�A�*

loss��:z�3�       �	r��ec�A�*

loss��G=E5�       �	)���ec�A�*

loss H�<\�/       �	G��ec�A�*

loss߄�<I��       �	���ec�A�*

loss�iE=���       �	Œ��ec�A�*

loss�Ծ<��T       �	�0��ec�A�*

loss��=B�       �	S͸�ec�A�*

loss��$<���       �	�y��ec�A�*

lossNO;ĉ��       �	���ec�A�*

lossc[F<ӡ �       �	����ec�A�*

loss�D�=y6�       �	U��ec�A�*

loss�R=�d       �	���ec�A�*

loss���=ȋ�A       �	ߋ��ec�A�*

loss�Fg=ң�,       �	�&��ec�A�*

loss#u;"�jQ       �	2ɽ�ec�A�*

loss�T�;2��       �	|`��ec�A�*

loss�ƭ<Аk�       �	r���ec�A�*

lossl@�<-|F       �	ѐ��ec�A�*

lossM
=�r�j       �	E+��ec�A�*

lossI��<(}��       �	z���ec�A�*

loss�'>{0�       �	W��ec�A�*

loss/*&< �       �	����ec�A�*

lossD�<�ne�       �	v���ec�A�*

loss���<b�N       �	 ��ec�A�*

lossS�<2g�e       �	^���ec�A�*

lossWN7<[�V       �	�Y��ec�A�*

lossnA�;��-       �	���ec�A�*

lossM�=qm�H       �	8���ec�A�*

lossF7=�t��       �	BB��ec�A�*

loss$P�=�a       �		���ec�A�*

loss�i�<o��!       �	0���ec�A�*

loss-��<zzW�       �	%A��ec�A�*

loss,(�;��:%       �	����ec�A�*

loss�=D��A       �	����ec�A�*

loss�<�=yܭ       �	[$��ec�A�*

lossn�<��       �	z���ec�A�*

lossV�=����       �	p_��ec�A�*

loss#��<��Bi       �	����ec�A�*

loss7l�=��C       �	,���ec�A�*

loss� <���        �	W��ec�A�*

loss6Dp=�2|       �	����ec�A�*

loss�T<��u       �	>���ec�A�*

loss�_�;�\1m       �	82��ec�A�*

loss�q�=͊/�       �	����ec�A�*

lossa��<�Y��       �	y��ec�A�*

loss*z6<��J�       �	���ec�A�*

loss���=����       �	?���ec�A�*

loss�V6;�       �	O��ec�A�*

loss*Z;�b       �	���ec�A�*

loss ӂ<���       �	h���ec�A�*

loss6��;|�w       �	�R��ec�A�*

lossm{F<���       �	����ec�A�*

loss׮�=��U<       �	.���ec�A�*

loss8��=�J�       �	�3��ec�A�*

loss�;y<��g       �	����ec�A�*

loss9;0%�       �	����ec�A�*

lossN�:��e�       �	J`��ec�A�*

lossv��<���       �	[���ec�A�*

lossj�<}�r�       �	�M��ec�A�*

loss�d�;���7       �	O���ec�A�*

loss���<�92�       �	���ec�A�*

loss��f<����       �	;��ec�A�*

losso��<��e�       �	ö��ec�A�*

lossV�=Aҕ8       �	�R��ec�A�*

loss7��=TB
Y       �	b���ec�A�*

loss�%O=��t       �	���ec�A�*

lossQ��<���        �	a2��ec�A�*

loss��<���       �	Q���ec�A�*

loss7;?�       �	%w��ec�A�*

lossx�	<�       �	���ec�A�*

loss���;îx�       �	ޫ��ec�A�*

loss��<�� h       �	�C��ec�A�*

loss�p;:�}�       �		���ec�A�*

loss��=����       �	�v��ec�A�*

lossFU5<���f       �	���ec�A�*

loss�Q�<I���       �	`���ec�A�*

loss���=޶�       �	sc��ec�A�*

loss���;-��       �	���ec�A�*

loss *M<.��3       �	���ec�A�*

loss��X=��       �	+L��ec�A�*

lossw��;f�y�       �	����ec�A�*

loss�dY=a=�]       �	ӈ��ec�A�*

loss��<C��       �	�#��ec�A�*

loss��u=b���       �	����ec�A�*

loss�Y}=#���       �	TU��ec�A�*

loss�oo=�}S�       �	J��ec�A�*

lossx4�=�`8R       �	4���ec�A�*

lossO��<\̲U       �	?��ec�A�*

loss3�v;E<�       �	F���ec�A�*

lossӞ�;[�x�       �	^���ec�A�*

loss��=��`       �	�z��ec�A�*

loss_w�=w��<       �	i8��ec�A�*

loss�,�=9D��       �	�*��ec�A�*

loss��=r�o.       �	�X��ec�A�*

loss[��<)|��       �	�m��ec�A�*

loss�T�<atr�       �	P��ec�A�*

loss���<��       �	�=��ec�A�*

loss���;?e�a       �	d���ec�A�*

loss �,<gd��       �	����ec�A�*

loss*ZS=P{�       �	�u��ec�A�*

loss/U�<c\�H       �	���ec�A�*

loss��N=��|M       �	�l��ec�A�*

loss�(=��O�       �	��ec�A�*

loss���<]A�-       �	�$��ec�A�*

loss5�=�^�       �	����ec�A�*

loss
�w<W��       �	����ec�A�*

loss���<(E%�       �	D0��ec�A�*

loss�#/<oI�       �	v �ec�A�*

loss���;�̑E       �	��ec�A�*

loss�W�<P�&       �	���ec�A�*

loss�s<%$9       �	Ef�ec�A�*

loss�Ю<A��o       �	�ec�A�*

lossǍ<B��p       �	��ec�A�*

losse�<WR�       �	���ec�A�*

loss�ض=
�z�       �	�%�ec�A�*

loss�8�;�$       �	D��ec�A�*

loss��<��<+       �	9^�ec�A�*

lossJr�;�Ag�       �	���ec�A�*

loss�C=��GM       �	���ec�A�*

loss�;[=9�       �	�&�ec�A�*

lossv-4=��O       �	G��ec�A�*

loss`��<S^       �	Z	�ec�A�*

loss��<`�(       �	�	�ec�A�*

loss�A<$q��       �	K�
�ec�A�*

loss]�"=c�^       �	2�ec�A�*

loss9S=�?�       �	q��ec�A�*

loss]:=���       �	�g�ec�A�*

loss��=�)�       �	���ec�A�*

loss��z<s.	       �	l��ec�A�*

loss�z�;iЎ`       �	�(�ec�A�*

losso�;�5�       �	���ec�A�*

loss���:��       �	�^�ec�A�*

loss��Q<��cR       �	@��ec�A�*

loss�{=�|�       �	���ec�A�*

loss!��<m�9       �	�2�ec�A�*

loss��=q��       �	���ec�A�*

lossƂ�<����       �	Ǆ�ec�A�*

lossqU�=�x[       �	��ec�A�*

loss��Y=�SH       �	���ec�A�*

loss���<"��       �	u�ec�A�*

loss�,h<ٲP       �	��ec�A�*

loss�;��jU       �	�H�ec�A�*

loss�k�</��g       �	Z��ec�A�*

loss]M�<"Oy�       �	�6�ec�A�*

loss�3*<'�       �	��ec�A�*

loss�� =�4��       �	wg�ec�A�*

loss[�2=a-�       �	.�ec�A�*

loss�=��{a       �	���ec�A�*

lossDYm=q%��       �	w,�ec�A�*

loss�$=}��X       �	&��ec�A�*

loss8��=̤��       �	�g�ec�A�*

loss|8�<�	       �	�	�ec�A�*

lossL=>j}�       �	��ec�A�*

loss��<�7`s       �	�@�ec�A�*

lossvV=�_#�       �	���ec�A�*

loss�0=kM�w       �	K��ec�A�*

lossw�o:*�3I       �	�� �ec�A�*

loss���<�K�       �	�*!�ec�A�*

losss�'<Cw�]       �	%�!�ec�A�*

loss��=�8��       �	�a"�ec�A�*

losss��<���       �	{�"�ec�A�*

loss��=wUr�       �	To$�ec�A�*

lossL	=��,1       �	1%�ec�A�*

loss�h�:K\ƥ       �	u�%�ec�A�*

loss��<0�0       �	DL&�ec�A�*

lossd�L;�{�b       �	 �&�ec�A�*

lossv�A;��A4       �	/�'�ec�A�*

lossw��=3�       �	�%(�ec�A�*

loss�R�;�]Sr       �	Q�(�ec�A�*

loss�>0;���b       �	_)�ec�A�*

loss�
b=vD�       �	 �)�ec�A�*

loss��=wF�       �	
�*�ec�A�*

loss��=�G��       �	y;+�ec�A�*

loss��k=�5��       �	��+�ec�A�*

lossA_�< ��       �	�n,�ec�A�*

lossé7=^��}       �	n�-�ec�A�*

loss/'<�2N       �	gG.�ec�A�*

lossz]=�So       �	�.�ec�A�*

lossIr=��tq       �	�{/�ec�A�*

loss�,N=R�J�       �	�>1�ec�A�*

loss��<��Ӵ       �	��1�ec�A�*

loss�d,==y��       �	5}2�ec�A�*

lossw�5<i `       �	W'3�ec�A�*

loss ^�=��g�       �	��3�ec�A�*

loss0.!=L���       �	�^4�ec�A�*

loss�Dg<tm�        �	A5�ec�A�*

loss��=2~re       �	T�5�ec�A�*

loss�q�< �9        �	fI6�ec�A�*

loss��>=�H��       �	��6�ec�A�*

lossi�];�7�       �	��7�ec�A�*

loss1Y�<�nL�       �	�8�ec�A�*

lossd5�<Y
l       �	b�8�ec�A�*

loss���<��>       �	aS9�ec�A�*

loss��<r���       �	��9�ec�A�*

loss�5�<��e�       �	�E;�ec�A�*

loss�`8=��@�       �	��;�ec�A�*

loss�O�<��~�       �	��<�ec�A�*

lossF>h�wL       �	/6=�ec�A�*

loss�va=.�Y[       �	��=�ec�A�*

loss)�f;q�        �	+j>�ec�A�*

loss�\ =x��C       �	�>�ec�A�*

loss��	=\�.       �	t�?�ec�A�*

loss&��<c���       �	'3@�ec�A�*

lossO�<w�"�       �	}�@�ec�A�*

loss��=غ�       �	fA�ec�A�*

loss:Zh<�<o�       �	m�A�ec�A�*

loss�f�<a�(Q       �	�B�ec�A�*

loss.�<���        �	�,C�ec�A�*

loss�v�;�b��       �	@�C�ec�A�*

lossF�l<M<�9       �	͕D�ec�A�*

loss���<��       �	BBE�ec�A�*

loss���;	�B#       �	��E�ec�A�*

lossD<��v       �	�zG�ec�A�*

loss@��=W���       �	<H�ec�A�*

loss�=M�ɕ       �	ʩH�ec�A�*

loss�ɸ;�]�       �	^�I�ec�A�*

lossʪ�<օ       �	q<J�ec�A�*

loss ��<dv�d       �	�J�ec�A�*

loss��r<րw       �	|K�ec�A�*

loss���<u͢V       �	:!L�ec�A�*

loss|�:=+���       �	�L�ec�A�*

loss�<�=���       �	ffM�ec�A�*

loss=��<v%�       �	��M�ec�A�*

loss!�<3�5�       �	��N�ec�A�*

losse�<��       �	�\O�ec�A�*

losscMP=���       �	�O�ec�A�*

lossC�:NɅ       �	@�P�ec�A�*

lossN��; Ƃ�       �	 RQ�ec�A�*

lossI^(<g���       �	W�Q�ec�A�*

losso=T�B       �	��R�ec�A�*

loss!�=��"I       �	�S�ec�A�*

lossC�<	~��       �	_�S�ec�A�*

loss];l<�=�       �	JT�ec�A�*

lossd%V=��$@       �	��T�ec�A�*

lossA�=�9 6       �	=�U�ec�A�*

loss0#<̋�       �	�+V�ec�A�*

loss(��=��4       �	�V�ec�A�*

loss���<��       �	�cW�ec�A�*

loss� =[9��       �	�X�ec�A�*

loss��-=3�       �	��X�ec�A�*

loss�@�<k+       �	73Y�ec�A�*

loss6D=�h�       �	��Y�ec�A�*

loss�a�=�o�       �	�`Z�ec�A�*

lossQ�=��@       �	�[�ec�A�*

loss�<9w�       �	��[�ec�A�*

lossi;�<���       �	H7\�ec�A�*

loss�<p=˦�       �	B�\�ec�A�*

loss��;d�       �	sh]�ec�A�*

loss�=\��       �	�^�ec�A�*

loss��p<ʂd       �	E�^�ec�A�*

loss-��<)�B       �	>?_�ec�A�*

loss�E�<ewm�       �	j�_�ec�A�*

loss�-�<�       �	�r`�ec�A�*

loss�=MQ�       �	|a�ec�A�*

lossT�;��*i       �	ߨa�ec�A�*

loss.�2=�6�       �	�@b�ec�A�*

losss�4=MH9Z       �	=�b�ec�A�*

lossn��<B�M�       �	�qc�ec�A�*

loss��>��f:       �	h d�ec�A�*

loss�; `��       �	ǻd�ec�A�*

loss4�J:9&��       �	\e�ec�A�*

lossN�=f%f       �	�f�ec�A�*

loss�8�<��H�       �	��f�ec�A�*

loss̯�<�m�       �	�Kg�ec�A�*

loss��~<�Q�%       �	�g�ec�A�*

loss�3&<?n�7       �	F�h�ec�A�*

loss�`=xG*       �	y=i�ec�A�*

lossݰ6<�F��       �	=�i�ec�A�*

loss*dy;���       �	mj�ec�A�*

loss&oM=I��       �	�k�ec�A�*

loss;�<       �	E�k�ec�A�*

lossoI>D��G       �	6l�ec�A�*

loss�2�:[ӗ4       �	(�l�ec�A�*

losss];$Xc       �	pm�ec�A�*

loss�`/=�        �	ln�ec�A�*

loss��=k�Z       �	Y�n�ec�A�*

loss<	<^P��       �	d:o�ec�A�*

lossҺ�=���       �	�o�ec�A�*

lossf�v=��b�       �	(dp�ec�A�*

loss{h;!H�       �	]�p�ec�A�*

loss��<(>D�       �	5�q�ec�A�*

loss,>=��o       �	�1r�ec�A�*

lossY�"=�i�6       �	1�r�ec�A�*

lossg3=��.m       �	�gs�ec�A�*

loss�F<�Oj^       �	n�s�ec�A�*

loss༩=D��E       �	��t�ec�A�*

lossT��<_�P�       �	g(u�ec�A�*

loss�iV=m�M�       �	/�u�ec�A�*

loss�D=3�3a       �	�uv�ec�A�*

loss�U�=(>�	       �	�w�ec�A�*

loss�+*=$I1�       �	��w�ec�A�*

loss�	�<uBe#       �	�Xx�ec�A�*

lossZ�;Y��       �	��x�ec�A�*

loss�O�=�Pz       �	l�y�ec�A�*

loss��p=��fk       �	�9z�ec�A�*

loss��i=Q��z       �	0�z�ec�A�*

lossΣ$< Q       �	,�{�ec�A�*

loss�n<)S�Q       �	.|�ec�A�*

loss���:җի       �	Y�|�ec�A�*

loss��*>5�}�       �	�W}�ec�A�*

loss'V�=�X�       �	%�}�ec�A�*

loss[`�<���[       �	��~�ec�A�*

loss�PF<�E.       �	�0�ec�A�*

loss��=����       �	C��ec�A�*

loss*a�<��V       �	�g��ec�A�*

lossAU�<U���       �	#k��ec�A�*

loss��=��       �	W��ec�A�*

lossX�p;
�W5       �	����ec�A�*

lossښW<B�E       �	�6��ec�A�*

loss9=z���       �	�̓�ec�A�*

loss�=
BM       �	�h��ec�A�*

loss�<ל��       �	���ec�A�*

loss�j�<�xk       �	����ec�A�*

loss_cc<�]�       �	6=��ec�A�*

loss��+<��$       �	�Ԇ�ec�A�*

loss�5=<B��       �	�l��ec�A�*

loss��=�+�       �	��ec�A�*

loss�	=���,       �	dɈ�ec�A�*

loss6=�<�       �	?��ec�A�*

loss���=���       �	���ec�A�*

loss!,q;�O�       �	�;��ec�A�*

loss�?�<���       �	Q؋�ec�A�*

loss���=�]|       �	�o��ec�A�*

loss�0�=�+&       �	���ec�A�*

loss���<q���       �	7���ec�A�*

loss�7<�F�       �	�:��ec�A�*

loss�E�=ԥ��       �	@܎�ec�A�*

lossD�.<���l       �	�{��ec�A�*

lossD�=���       �	���ec�A�*

loss�/�:�7�?       �	����ec�A�*

loss�$o<��-       �	iR��ec�A�*

lossF��<B��9       �	���ec�A�*

loss�΃<h�       �	t|��ec�A�*

loss��J<Ĩ�       �	��ec�A�*

loss�6�<AcZf       �	\���ec�A�*

loss�~Y<N��H       �	�?��ec�A�*

loss���;G�|�       �	5Ҕ�ec�A�*

lossX�C<�F�       �	�f��ec�A�*

loss�§;���       �	����ec�A�*

loss�	B=G��       �	J���ec�A�*

loss[8x<�c��       �	�-��ec�A�*

lossMx�;V��m       �	�ŗ�ec�A�*

loss]�y;[�~�       �	:]��ec�A�*

lossD��;�i�       �	"��ec�A�*

loss��P;�O       �	����ec�A�*

loss�;v�       �	�C��ec�A�*

loss.��=bL�       �	+���ec�A�*

loss�9&;��{       �	����ec�A�*

loss�y<4ϳz       �	���ec�A�*

loss���=70j       �	Ĵ��ec�A�*

loss1ư;�bu~       �	oK��ec�A�*

loss�=�,       �	���ec�A�*

loss��S;2�V_       �	o���ec�A�*

lossɻ$<'A~�       �	4��ec�A�*

lossq�; u]       �	Sʟ�ec�A�*

loss�@=d3�       �	Lp��ec�A�*

loss_��=0mv       �	�y��ec�A�*

loss��>�Й�       �	���ec�A�*

loss
�`<b�.�       �	W���ec�A�*

loss���;yy��       �	�L��ec�A�*

loss�5q=�%�M       �	����ec�A�*

loss�v<�
�T       �	���ec�A�*

lossT��;)���       �	F\��ec�A�*

loss�{ =��^       �	����ec�A�*

lossݧ~<���:       �	>���ec�A�*

loss�j�=���       �	�(��ec�A�*

loss�|<u��K       �	�ħ�ec�A�*

loss׬�=C�T�       �	�`��ec�A�*

loss�р=8_^O       �	���ec�A�*

loss�\ =d"x       �	��ec�A�*

loss[/�;B���       �	�&��ec�A�*

loss�V%<ﱤr       �	9Ԫ�ec�A�*

loss�~,<�</X       �	�x��ec�A�*

loss�	=���       �	��ec�A�*

loss1g�<�7H        �	����ec�A�*

losst�\<~4       �	g`��ec�A�*

lossI��<H�&�       �	���ec�A�*

loss��G;B�ڃ       �	d���ec�A�*

loss�s�;Ɣ�O       �	1��ec�A�*

loss�zU=k�"�       �	�ͯ�ec�A�*

loss���;���       �	@k��ec�A�*

loss�=���       �	9^��ec�A�*

lossD;�<ߗ�       �	@���ec�A�*

loss��1;��       �	⑲�ec�A�*

loss�q�;��Y       �	�.��ec�A�*

loss��|<#6!       �	j³�ec�A�*

loss8�[=mԖy       �	�W��ec�A�*

lossN(i=ls�       �	���ec�A�*

loss�N�;�q�       �	+���ec�A�*

loss�ȯ;����       �	���ec�A�*

loss=��<h�ؼ       �	-���ec�A�*

loss�B=�Tph       �	0K��ec�A�*

lossvk=�R��       �	���ec�A�*

lossv2;�]B�       �	u��ec�A�*

loss7��<bL�       �	H��ec�A�*

loss���<���!       �	Ժ��ec�A�*

lossh�t:`�3b       �	�\��ec�A�*

loss��$=���       �	b���ec�A�*

loss��:!�a�       �	����ec�A�*

loss�\�<��A�       �	�*��ec�A�*

loss���=�&3       �	ļ�ec�A�*

lossD*�:i�e�       �	���ec�A�*

lossSz�<�c8       �	x%��ec�A�*

loss.�P=�ZA       �	�Ǿ�ec�A�*

loss��=�K{.       �	�ÿ�ec�A�*

lossx�3=�I�G       �	.Y��ec�A�*

loss��<T��       �	���ec�A�*

loss��i<I(�       �	֨��ec�A�*

loss݇]<bR�       �	!s��ec�A�*

loss��:�0�       �	{��ec�A�*

losswZ�<���       �	q���ec�A�*

loss��*:X]��       �	�E��ec�A�*

loss�l�;B�|       �	���ec�A�*

lossXj�<��*�       �	ZG��ec�A�*

loss�J;��Q�       �	;���ec�A�*

loss�:6r�       �	5y��ec�A�*

lossH�<���l       �	Y��ec�A�*

loss�8Z�=�       �	&���ec�A�*

loss��z9|b��       �	�w��ec�A�*

loss�<��       �	+��ec�A�*

lossD�<�aB       �	#���ec�A�*

loss��<wG�f       �	;T��ec�A�*

loss��w:��c       �	����ec�A�*

loss,;k��       �	����ec�A�*

loss�U=t���       �	.��ec�A�*

loss��m;A�nQ       �	����ec�A�*

lossl�=D+�       �	�a��ec�A�*

lossL�=-y��       �	����ec�A�*

lossNK<�v�K       �	����ec�A�*

lossND'<�W��       �	�&��ec�A�*

loss�n�<�@E       �	Ǻ��ec�A�*

lossrT;���       �	�P��ec�A�*

loss�ʩ=��p       �	����ec�A�*

loss<<\       �	8���ec�A�*

loss��.=�^ʓ       �	U.��ec�A�*

loss�t�;;�?�       �	���ec�A�*

lossz�>���[       �	�l��ec�A�*

loss@�y<��A       �	\��ec�A�*

lossf̠<K�/�       �	���ec�A�*

loss;ް<�µ       �	�B��ec�A�*

loss��<.1/       �	0���ec�A�*

losszQ=p�)C       �	Ym��ec�A�*

loss:��<��ſ       �	1��ec�A�*

lossjik=ÄP%       �	]���ec�A�*

loss}�<<��[       �	D��ec�A�*

loss��7;��{\       �	~7��ec�A�*

loss=�<��'�       �	g���ec�A�*

loss��<��       �	�o��ec�A�*

loss�=��l       �	M��ec�A�*

loss%�<�-�       �	����ec�A�*

loss�E�<E��       �	F\��ec�A�*

loss_��=��k       �	%��ec�A�*

loss��<
؝�       �	���ec�A�*

lossN�i<�Ob       �	�6��ec�A�*

loss���<���       �	R���ec�A�*

loss���<\8>1       �	�x��ec�A�*

loss!G@<��h       �	5��ec�A�*

loss/�;��       �	J���ec�A�*

loss��;�"�       �	����ec�A�*

loss�Ĩ;�m        �	O$��ec�A�*

lossto-;.�x       �	Q���ec�A�*

loss&.�<��8       �	yW��ec�A�*

loss��/<�{�       �	����ec�A�*

loss�g=�ͨ$       �	����ec�A�*

loss�xQ<�1d       �	�@��ec�A�*

lossfui<��       �	!���ec�A�*

loss}1t=�sp       �	����ec�A�*

loss���<0k�n       �	�1��ec�A�*

loss�<'��m       �	����ec�A�*

loss3iF=3_��       �	����ec�A�*

lossZ�=��S       �	
i��ec�A�*

loss�2�=��M�       �	
��ec�A�*

lossֲ<��aT       �	���ec�A�*

loss&.D=��P�       �	WA��ec�A�*

loss�*v;�J�p       �	j���ec�A�*

loss1�<T��       �	A���ec�A�*

loss@&<=#|$       �	�+�ec�A�*

loss0�=��<       �	��ec�A�*

losskX�<�M�E       �	���ec�A�*

loss���;�kWr       �	}$�ec�A�*

loss�;��\�       �	���ec�A�*

lossw�`<NtF�       �	�}	�ec�A�*

loss�I�<L#:�       �	�
�ec�A�*

loss��<i��       �	��
�ec�A�*

loss
#=��n�       �	�]�ec�A�*

lossr�<H(��       �	��ec�A�*

loss�4<\�8.       �	��ec�A�*

loss4N<OT�w       �	ZH�ec�A�*

loss%�<��|       �	Z��ec�A�*

loss���<��N�       �	5��ec�A�*

loss�ƾ<��(       �	�>�ec�A�*

lossq� =�A#�       �	���ec�A�*

loss`�[;nv�       �	y�ec�A�*

loss��1<�Ho�       �	3�ec�A�*

loss8E�<���       �	���ec�A�*

lossz�F=�o�       �	�b�ec�A�*

lossc�<��?�       �	�ec�A�*

loss�=�͚_       �	V��ec�A�*

loss4;BJ�g       �	�7�ec�A�*

loss?;R=�W�.       �	E��ec�A�*

lossb=r�@       �	j�ec�A�*

lossq��;���R       �	��ec�A�*

loss� <_ND�       �	l��ec�A�*

losso�<�l       �	�(�ec�A�*

loss�f�<����       �	,��ec�A�*

loss__<P�z�       �	M�ec�A�*

losszۀ;m`��       �	]��ec�A�*

lossA8=uZt       �	=�ec�A�*

loss�;ͦ�@       �	@�ec�A�*

lossl��<�E�N       �	O��ec�A�*

loss�u;�?       �	I�ec�A�*

lossHw�=�ΰ�       �	}��ec�A�*

loss�N;K���       �	τ�ec�A�*

loss��-<��~       �	��ec�A�*

loss$!�=�Ū       �	ϻ�ec�A�*

loss@tx=L�q       �	;T�ec�A�*

loss�7�;�3�       �	�ec�A�*

loss�*�;�i��       �	'��ec�A�*

loss;� ;�f�       �	�o �ec�A�*

lossZdP<d+�H       �	1!�ec�A�*

loss�W.=�;�f       �	{�!�ec�A�*

loss�l=�,T       �	�:"�ec�A�*

loss�Fg<�Uj       �	��"�ec�A�*

loss�.�<g�<�       �	��#�ec�A�*

loss���<u'Х       �	h $�ec�A�*

loss=$�:��,�       �	�$�ec�A�*

loss���<���-       �	GV%�ec�A�*

lossF�=,>       �	�&�ec�A�*

loss��F=y�8�       �	�&�ec�A�*

loss%;>�-eq       �	H'�ec�A�*

loss���<>�Q�       �	��'�ec�A�*

loss��i<�[{D       �	V(�ec�A�*

loss��=@       �	e)�ec�A�*

loss/<��y       �	6�)�ec�A�*

loss|��<�6M_       �	^g*�ec�A�*

loss�)=�/�u       �	+�ec�A�*

loss���<�P+       �	��+�ec�A�*

loss�ߏ:CRP<       �	`<,�ec�A�*

loss
(�;$��       �	��,�ec�A�*

loss��-=8B�       �	��-�ec�A�*

loss���<��o       �	@�.�ec�A�*

lossm��<��G�       �	��/�ec�A�*

lossV��=��       �	�z0�ec�A�*

loss��=���       �	�I1�ec�A�*

loss���=�[�       �	��1�ec�A�*

loss��=��{       �	|�2�ec�A�*

loss���;xC��       �	\�3�ec�A�*

loss��<8	�N       �	�L4�ec�A�*

lossڄ�<d�M�       �	�F5�ec�A�*

loss��R;�xp}       �	��5�ec�A�*

lossr��<�j�       �	��6�ec�A�*

loss�=pq8�       �	��7�ec�A�*

loss
S<~       �	\W8�ec�A�*

loss�b�;�{�       �	�9�ec�A�*

loss���<�K��       �	�:�ec�A�*

losssv�<-窑       �	��:�ec�A�*

loss���< �m       �	Ѯ;�ec�A�*

loss��<�|��       �	�t<�ec�A�*

lossT^w= Ҫ        �	;T=�ec�A�*

loss�9)=�7�       �	�>�ec�A�*

loss�$�<����       �	<�>�ec�A�*

lossL�<ԋ#       �	�s?�ec�A�*

loss�L<z<3[       �	d\@�ec�A�*

lossa�-<�xX       �	n�@�ec�A�*

lossS��<X��       �	pB�ec�A�*

loss�5�<���
       �	�B�ec�A�*

loss��<���3       �	�{C�ec�A�*

loss �*<Z��d       �	fD�ec�A�*

loss��~<L�;.       �	��D�ec�A�*

loss��4=�0�       �	�IE�ec�A�*

loss�]�<���r       �	.�E�ec�A�*

lossE<<�=       �	M�F�ec�A�*

losst��=���r       �	JG�ec�A�*

loss�8q=�V�       �	��G�ec�A�*

loss�l�=^�:�       �	�H�ec�A�*

loss|�;�?�s       �	��I�ec�A�*

loss�<SR9       �	��J�ec�A�*

loss�h5;�Ѷ       �	��K�ec�A�*

loss37�<}6s�       �	DOL�ec�A�*

loss=(t;?Z�       �	��L�ec�A�*

loss�7=,70�       �	�M�ec�A�*

loss��<��C�       �	��N�ec�A�*

loss��!=�OY�       �	0�O�ec�A�*

loss�7 =���       �	ƉP�ec�A�*

lossmH�<nAx       �	�0Q�ec�A�*

lossA�=g
�       �	�!R�ec�A�*

loss)2=N��       �	]mS�ec�A�*

loss��<,�*�       �	�bT�ec�A�*

loss���<�Y�$       �	�U�ec�A�*

lossQ$�;��       �	��U�ec�A�*

loss24'=Li�i       �	��V�ec�A�*

lossO�<f K       �	�9W�ec�A�*

loss��+<k���       �	
�W�ec�A�*

loss}�K<�KS       �	(X�ec�A�*

loss�z�<n�
       �	�+Y�ec�A�*

loss�W�=�$�O       �	@�Y�ec�A�*

loss�3�;��m�       �	�Z�ec�A�*

lossv׬<�sF       �	$[�ec�A�*

loss�<@|�<       �	J�[�ec�A�*

loss��<�uL�       �	Ts\�ec�A�*

lossH�	;%��b       �	n]�ec�A�*

lossc�=�v��       �	��]�ec�A�*

loss]�=<���       �	LP^�ec�A�*

loss�-S<�B�       �	��^�ec�A�*

lossr~<K��       �	�_�ec�A�*

loss��<5\�       �	`#`�ec�A�*

loss@g<p�<�       �	7Qa�ec�A�*

loss�g�<z	�       �	�,b�ec�A�*

loss��6<^�ͽ       �	��b�ec�A�*

loss�'p<��S�       �	mtc�ec�A�*

loss ��=��       �	�d�ec�A�*

loss�U�;���       �	ıd�ec�A�*

loss<��;��+�       �	�Te�ec�A�*

loss��J<��K�       �	��e�ec�A�*

loss�@�<-K��       �	�f�ec�A�*

lossd�,<6���       �	�*g�ec�A�*

loss��
;�2�       �	u�g�ec�A�*

loss!��<�;�-       �	gh�ec�A�*

loss<�	=�TF       �	�i�ec�A�*

loss=6<��       �	�3j�ec�A�*

lossA��<�Oե       �	��j�ec�A�*

loss��s=g_2       �	Dkk�ec�A�*

loss/M�<�<oa       �	�l�ec�A�*

loss$�;k}Z�       �	P�l�ec�A�*

loss�Ո;)��       �	�Em�ec�A�*

lossf��;F]�}       �	z�m�ec�A�*

lossM�;���       �	�xn�ec�A�*

loss�u�<R�&�       �	y:o�ec�A�*

loss�N==T��       �	�wp�ec�A�*

loss��=.�)�       �	�q�ec�A�*

lossZ=" �       �	�r�ec�A�*

loss��;�<c       �	��r�ec�A�*

loss�y�;]��       �	Ήs�ec�A�*

loss�� >R C       �	st�ec�A�*

lossM��=�p@�       �	�du�ec�A�*

lossl"�;Jf�J       �	Yv�ec�A�*

loss[�]<#Su�       �	H�v�ec�A�*

loss)U4<�� /       �	�w�ec�A�*

loss���<N={Z       �	{2x�ec�A�*

loss�<P��       �	��x�ec�A�*

loss��
>햆�       �	sey�ec�A�*

loss�sf<}�        �	��y�ec�A�*

loss�;N��A       �	��z�ec�A�*

loss�D=�W0       �	'{�ec�A�*

loss�{e<;�"       �	�{�ec�A�*

loss�K<�D��       �	yu|�ec�A�*

lossO�;�T�Q       �	�o}�ec�A�*

loss!��;�`�       �	�~�ec�A�*

loss8�<�8\       �	��~�ec�A�*

loss1��<�+X"       �	9D�ec�A�*

loss���<�KS       �	Q��ec�A�*

lossV� =��Lq       �	?o��ec�A�*

loss�"�=(5�~       �	���ec�A�*

loss�l"=���       �	���ec�A�*

loss�#�=T�%       �	a���ec�A�*

loss�;*E��       �	� ��ec�A�*

loss��9���<       �	e�ec�A�*

loss@�R=;w0�       �	�_��ec�A�*

lossJ~�<���W       �		��ec�A�*

loss�g;X�N�       �	���ec�A�*

loss�a�<��#�       �	E��ec�A�*

loss�k;O���       �	�a��ec�A�*

loss�&=
���       �	 ��ec�A�*

loss��=E�fZ       �	����ec�A�*

loss�_�<�$�       �	����ec�A�*

loss�~�<�G3f       �	����ec�A�*

loss�#=�pG�       �	��ec�A�*

loss2��<����       �	F#��ec�A�*

loss���<�jC
       �	)ό�ec�A�*

loss�.�<�Jc,       �	;s��ec�A�*

loss/��<���       �	���ec�A�*

loss���;�/y�       �	�Ǝ�ec�A�*

loss�k;�iE       �	�q��ec�A�*

loss��=���       �	w��ec�A�*

loss�<U6�       �	��ec�A�*

lossa�Y<	G��       �	�i��ec�A�*

loss]1V=0�ז       �	���ec�A�*

loss8��;��       �	>���ec�A�*

loss�+=�E�       �	g_��ec�A�*

loss���=����       �	+���ec�A�*

loss���<l�       �	`��ec�A�*

loss�>�=F}�b       �	���ec�A�*

lossp��=���       �	���ec�A�*

loss�DE=�d�       �	�<��ec�A�*

loss�B�<�Ƨ7       �	�ߗ�ec�A�*

loss��N<��^"       �	6x��ec�A�*

loss�h<�	b�       �	Q��ec�A�*

loss��O<���       �	y���ec�A�*

loss샎<)�h       �	{O��ec�A�*

loss�u�<���       �	)��ec�A�*

loss�U!=�Pz�       �	���ec�A�*

lossQ��=k��x       �	�'��ec�A�*

loss?��=�|�B       �	����ec�A�*

loss��>V6W       �	�d��ec�A�*

lossX=E��       �	����ec�A�*

loss��"=�*�       �	=���ec�A�*

lossW�-=���       �	�5��ec�A�*

lossO��;;�       �	ϟ�ec�A�*

lossn�<ooVG       �	v��ec�A�*

loss�>�=f�       �	� ��ec�A�*

loss���;��},       �	Iڡ�ec�A�*

loss�T/<����       �	4���ec�A�*

loss*Y�<��L       �	t#��ec�A�*

loss��p<���:       �	�£�ec�A�*

loss�+�<�<I       �	@j��ec�A�*

loss��y<=��F       �	���ec�A�*

loss���;�3��       �	����ec�A�*

loss�Ƞ; ���       �	NB��ec�A�*

loss��d=?7U3       �	���ec�A�*

loss�O<�HX�       �	�{��ec�A�*

lossW��<�eV       �	^��ec�A�*

loss�[5<:��       �	h���ec�A�*

loss�n�<�ɍY       �	�U��ec�A�*

lossZe,<N:��       �	(��ec�A�*

loss�&�<�"�       �	.���ec�A�*

loss!mj;@��?       �	D3��ec�A�*

loss9�;���L       �	�̫�ec�A�*

losswLq=�+C       �	�o��ec�A�*

loss] x;Y��       �	�!��ec�A�*

lossw�==���>       �	����ec�A�*

loss=�<x�7�       �	\W��ec�A�*

loss���<L�       �	 ��ec�A�*

loss��<a�J�       �	࠯�ec�A�*

loss���<&l\�       �	�A��ec�A�*

lossȖG=�2��       �	Lݰ�ec�A�*

loss��^<�)-�       �	՗��ec�A�*

lossLe =��M�       �	I0��ec�A�*

loss�]e<'O��       �	�̲�ec�A�*

loss�[=mO�
       �	g��ec�A�*

loss��;�H3       �	����ec�A�*

loss�V�;�{r       �	���ec�A�*

loss�Ǖ;���       �	l���ec�A�*

lossϢv<m�W       �	�+��ec�A�*

loss#y=�
܇       �	Kʶ�ec�A�*

loss3ow<'О       �	�b��ec�A�*

lossO�Q=���       �	3���ec�A�*

loss�|�=%�M       �	Ւ��ec�A�*

loss��=�8       �	J^��ec�A�*

loss��;��B2       �	����ec�A�*

loss)��;_���       �	���ec�A�*

loss�kv=1 0�       �	�'��ec�A�*

loss26[;%"-O       �	w���ec�A�*

loss�B�<���       �	uV��ec�A�*

loss��=��SV       �	 ��ec�A�*

loss��{=n�-       �	����ec�A�*

loss�'�<q�ޠ       �	q��ec�A�*

lossh�:6��C       �	����ec�A�*

loss��;bT�       �	�N��ec�A�*

loss�=\.wa       �	M��ec�A�*

lossj��<�C�       �	����ec�A�*

lossNT�<K5��       �	}>��ec�A�*

loss��{=��8K       �	���ec�A�*

loss=U�<�W�p       �	_|��ec�A�*

loss(F;�3��       �	��ec�A�*

lossv=�2{t       �	����ec�A�*

lossEhW=��r�       �	C��ec�A�*

loss\`�;�T��       �	����ec�A�*

loss�;!�Ƿ       �	qr��ec�A�*

lossR@�<����       �	#N��ec�A�*

loss��;�$-       �	���ec�A�*

loss1(]=��       �	O���ec�A�*

loss=7�<�[�7       �	$H��ec�A�*

loss:�-=�R.       �	����ec�A�*

lossv�{;�0       �	)y��ec�A�*

lossٍ=I�Wn       �	�[��ec�A�*

loss���<=B
       �	s���ec�A�*

lossNt;�U=`       �	G���ec�A�*

lossa	�<w��H       �	�(��ec�A�*

loss��{;sG       �	���ec�A�*

loss�ț:m�K�       �	�Z��ec�A�*

loss@�<�֙�       �	����ec�A�*

loss��=�X�       �	����ec�A�*

lossF�2<EA       �	=,��ec�A�*

lossq;�=$�e       �	����ec�A�*

losshi=��       �	[x��ec�A�*

lossQ �=�7/�       �	����ec�A�*

lossԼz;��q�       �	a��ec�A�*

loss���;��t�       �	����ec�A�*

loss��;�x�x       �	����ec�A�*

loss���<8a��       �	�8��ec�A�*

loss�Z<�}�H       �	F���ec�A�*

loss�]=�p�F       �	{f��ec�A�*

loss8G6;X�C       �	����ec�A�*

loss��=�q8       �	g���ec�A�*

losszN<�"��       �	�1��ec�A�*

loss9+<GG�m       �	m���ec�A�*

loss���;pEu       �	�e��ec�A�*

loss���<G�%       �	���ec�A�*

loss���<���o       �	(���ec�A�*

loss��8<��p       �	�?��ec�A�*

loss��S<��]V       �	����ec�A�*

loss���=S�$�       �	-���ec�A�*

lossT^I<�~�Q       �	�4��ec�A�*

loss
�<d���       �	����ec�A�*

loss@ϡ<`�n       �	�g��ec�A�*

loss�}�=����       �	?��ec�A�*

loss�kG=4K�*       �	���ec�A�*

loss�`M=�Q�N       �	�1��ec�A�*

loss�po<��       �	����ec�A�*

loss�:�<�
\       �	�o��ec�A�*

lossi��<�N�       �	���ec�A�*

loss\B=Y��       �	���ec�A�*

lossO#q=�,�       �	�2��ec�A�*

loss��<�,Ӝ       �	R���ec�A�*

lossd�<�o�H       �	Ui��ec�A�*

loss�4<�u��       �	j���ec�A�*

lossݑG=B �       �	���ec�A�*

lossO~<t�^       �	n1��ec�A�*

loss�[9<Pk�W       �	}���ec�A�*

loss	�>=t��       �	R`��ec�A�*

loss���<_��W       �	Y���ec�A�*

loss��;C�T�       �	����ec�A�*

loss��=��,z       �	g,��ec�A�*

loss�=�<��~�       �	9���ec�A�*

loss==�#�       �	zp��ec�A�*

loss���;_p��       �	���ec�A�*

loss�f;�x�       �	����ec�A�*

loss�_<G_��       �		6��ec�A�*

loss|�=MR~       �	����ec�A�*

loss�̶<f*R       �	q��ec�A�*

lossn?2<��<�       �	���ec�A�*

loss�Q=�~       �	����ec�A�*

loss�D;���       �	{L��ec�A�*

losse�w=^�H�       �	-���ec�A�*

loss�|U=̦0�       �	Ҏ��ec�A�*

loss���<, Oz       �	�w��ec�A�*

loss���<59{�       �	���ec�A�*

loss��;���       �	����ec�A�*

loss�L�=Y>#�       �	���ec�A�*

loss3�-=����       �	W���ec�A�*

loss�)=o�       �	����ec�A�*

loss��;��\       �	$b��ec�A�*

loss�@=�K�       �	��ec�A�*

lossH�)<7�l�       �	?��ec�A�*

loss$f�<	�"       �	����ec�A�*

loss�w7<°<       �	w���ec�A�*

loss͒S<��v       �	35��ec�A�*

loss��<�WL"       �	����ec�A�*

loss��U<����       �	
i��ec�A�*

lossW�S=���g       �	���ec�A�*

lossȬ�;Q��k       �	R���ec�A�*

lossd�0<0L��       �	�5��ec�A�*

lossi/c=R+��       �	6���ec�A�*

loss�c<�ԟ       �	:u��ec�A�*

loss�d/<V�[!       �	��ec�A�*

loss�VR=#~�       �	N���ec�A�*

lossWC�<�X�       �	�N��ec�A�*

lossy�<m�v�       �	����ec�A�*

loss�6'=a�h�       �	�� �ec�A�*

loss=z�<�pε       �	r5�ec�A�*

loss$
?<�^�        �	5��ec�A�*

loss�fb;�K�       �	�l�ec�A�*

lossWm=k$"�       �	b�ec�A�*

loss&X�;P��>       �	y��ec�A�*

lossNǼ;�_
c       �	%A�ec�A�*

lossR>=�
J�       �	;��ec�A�*

lossƜ=<A�J?       �	{�ec�A�*

losss'�<�{��       �	��ec�A�*

lossZ>=Z�ì       �	���ec�A�*

loss���:����       �	PR�ec�A�*

loss��;���       �	���ec�A�*

lossW:�<}kw       �	��ec�A�*

loss:h�<1���       �	�3	�ec�A�*

loss�;t=f丹       �	*�	�ec�A�*

lossM<��
       �	�_
�ec�A�*

loss$�/<ɲ&E       �	j�
�ec�A�*

lossE<���~       �	͏�ec�A�*

loss���;h�,%       �	�&�ec�A�*

lossf��;��#       �	+��ec�A�*

loss�<ΐ�8       �	*U�ec�A�*

loss���;b�پ       �	���ec�A�*

lossEo<1�H       �	��ec�A�*

loss�g:��[#       �	��ec�A�*

lossV�B<p���       �	_��ec�A�*

loss�~<z��       �	�K�ec�A�*

loss��;���       �	���ec�A�*

loss?"�<�O��       �	}x�ec�A�*

loss���<��J       �	G!�ec�A�*

loss��<��C       �	���ec�A�*

loss߀�=0�X�       �	�S�ec�A�*

loss�'=VyW       �	��ec�A�*

lossQ�<\Y��       �	R��ec�A�*

loss�
(=��=w       �	�1�ec�A�*

loss;�;ws�       �	u��ec�A�*

loss{�<�1�       �	�]�ec�A�*

lossk�<�"1       �	��ec�A�*

lossJ�<\E�       �	`��ec�A�*

lossl
= �O       �	{J�ec�A�*

loss�G;=�t��       �	2��ec�A�*

loss@}�<�['�       �	c{�ec�A�*

loss�O�<�υ$       �	�ec�A�*

lossɩ�;h+ɸ       �	��ec�A�*

loss+�;?q�       �	�N�ec�A�*

loss��;d���       �	��ec�A�*

loss%�<�1��       �	���ec�A�*

loss; ;=7#�       �	4�ec�A�*

loss��;����       �	��ec�A�*

lossX��;M��       �	n��ec�A�*

loss��;�03�       �	g�!�ec�A�*

loss�/3=�E�       �	~8"�ec�A�*

loss.�0=�Y�       �	��"�ec�A�*

loss̕<�u�w       �	V�#�ec�A�*

loss��;�a       �	>&$�ec�A�*

lossC�<�?X�       �	g,%�ec�A�*

loss��~<�F(�       �	�%�ec�A�*

loss/�<|���       �	)Y&�ec�A�*

loss#k
=��B       �	_�&�ec�A�*

loss7�<����       �	��'�ec�A�*

lossN\�:x�       �	�((�ec�A�*

lossXe<��Jq       �	P�(�ec�A�*

loss�[�;/-�       �	�`)�ec�A�*

loss�2�;=�/�       �	��)�ec�A�*

loss�O�<}�1L       �	��*�ec�A�*

losst�<
(5K       �	&7+�ec�A�*

lossq˝:�ms       �	��+�ec�A�*

loss���<��D       �	�,�ec�A�*

loss;�f=Vb�       �	(-�ec�A�*

loss@�<�6�       �	B�-�ec�A�*

loss��\=S?o       �	iq.�ec�A�*

lossv@D=�I��       �	�/�ec�A�*

loss��:Y��       �	��/�ec�A�*

loss�m;-u��       �	��0�ec�A�*

loss��=
�	I       �	�+1�ec�A�*

lossC[�<>p�O       �	S�1�ec�A�*

losszʒ<��Ֆ       �	3j2�ec�A�*

loss�$<Lφ�       �	�3�ec�A�*

loss�#�<��        �	ܛ3�ec�A�*

loss\��;F��       �	�54�ec�A�*

loss}��<��t�       �	��4�ec�A�*

lossn�D=$z\$       �	�5�ec�A�*

loss�+<ι�       �	�-6�ec�A�*

lossF�<��"^       �	��6�ec�A�*

lossS��<5�}�       �	\t7�ec�A�*

loss,�<���       �	�8�ec�A�*

loss�F9<C8�       �	��8�ec�A�*

lossZ�
=i�
        �	�Z9�ec�A�*

lossсz<���       �	#�9�ec�A�*

loss���;Y��       �	��:�ec�A�*

loss[�;�g�       �	JA;�ec�A�*

loss�q�<t�X�       �	��;�ec�A�*

loss�
�<F��.       �	Z<�ec�A�*

loss\$�<�p�       �	c(=�ec�A�*

loss�Y�<���       �	x�=�ec�A�*

loss�D;�=��       �	R}>�ec�A�*

losst)!<�V�(       �	-%?�ec�A�*

loss�,+<py��       �	��?�ec�A�*

loss���=_���       �	�n@�ec�A�*

loss��;^ἅ       �	A�ec�A�*

lossX�t;��%       �	T�A�ec�A�*

loss!�=�d̢       �	�DB�ec�A�*

loss<t<�LҀ       �	��B�ec�A�*

loss�q<��I�       �	�C�ec�A�*

loss��Q;�7�       �	�%D�ec�A�*

loss��P<��.�       �	a�D�ec�A�*

lossZ�'=�!       �	�cE�ec�A�*

lossr��={<=       �	 F�ec�A�*

lossJ�H<<jxb       �	�G�ec�A�*

lossT�;<�tҳ       �	��G�ec�A�*

lossԝk<�$@�       �	K�H�ec�A�*

lossO�=�2�       �	'lI�ec�A�*

loss%�3=Yo��       �	�J�ec�A�*

loss\
V<g��        �	P�K�ec�A�*

loss�Dz<�)��       �	�=L�ec�A�*

loss+)�=:�#       �	!sM�ec�A�*

loss�X=�u�       �	�8N�ec�A�*

lossd�<�ҮV       �	�N�ec�A�*

loss��=+<�J       �	=�O�ec�A�*

loss�KT<+���       �	7�P�ec�A�*

loss��;'۸�       �	�cQ�ec�A�*

loss&=z;I!��       �	��Q�ec�A�*

lossc�<��ԃ       �	% S�ec�A�*

lossev<=��b�       �	(�S�ec�A�*

loss7˧<޴��       �	��T�ec�A�*

lossCf�<��h�       �	YU�ec�A�*

lossxx}<|�-       �	j�U�ec�A�*

loss�6=-��       �	ސV�ec�A�*

loss��=-��       �	3W�ec�A�*

loss���:Y.ڧ       �	��W�ec�A�*

loss
D<P�k5       �	ƉX�ec�A�*

loss3�o;5�N�       �	�(Y�ec�A�*

loss�8M=FKT       �	C�Y�ec�A�*

loss[��<L�N       �	#iZ�ec�A�*

loss���<7�M�       �	�[�ec�A�*

loss��;[�`       �	�[�ec�A�*

loss��<�9%       �	�8\�ec�A�*

loss�;����       �	��\�ec�A�*

loss�Y<B���       �	�m]�ec�A�*

loss��=;}$�       �	^�ec�A�*

loss�+<�ӵ       �	�^�ec�A�*

lossư�=�)�       �	�x_�ec�A�*

lossl$l<F�WP       �	�`�ec�A�*

loss,�=Da       �	?�`�ec�A�*

loss�0<m �       �	�Ea�ec�A�*

loss+��=�O��       �	�a�ec�A�*

loss*�<�}(�       �	zrb�ec�A�*

loss �<	%i1       �	�c�ec�A�*

loss,��;^Xg       �	��c�ec�A�*

losst<K        �	CUd�ec�A�*

loss8�:�֓�       �	��d�ec�A�*

loss�@&<�H,       �	��e�ec�A�*

lossr�= d��       �	;f�ec�A�*

loss���;{�y       �	��f�ec�A�*

lossf��<� �       �	<Mg�ec�A�*

loss��<�ܤ       �	�g�ec�A�*

loss]/A=��0       �	�h�ec�A�*

lossʹ�;mi��       �	�#i�ec�A�*

losss|�<��       �	�i�ec�A�*

loss��>���4       �	�bj�ec�A�*

loss&��;�G_�       �	�k�ec�A�*

loss���:�7�       �	*�k�ec�A�*

lossM/=�}o�       �	�@l�ec�A�*

loss�dQ;I�=       �	H�l�ec�A�*

loss�<I��W       �	�vm�ec�A�*

loss���;	�       �	An�ec�A�*

loss���:a/J�       �	��n�ec�A�*

loss�:I;�MƳ       �	'Mo�ec�A�*

loss��p=;#�       �	[�o�ec�A�*

loss��9�ą       �	��p�ec�A�*

loss���9�8I       �	�2r�ec�A�*

loss���9�7�=       �	+1s�ec�A�*

lossk��</!��       �	6�s�ec�A�*

loss�D�<nzvl       �	�kt�ec�A�*

loss��_9��'�       �	�u�ec�A�*

loss]!<����       �	 �u�ec�A�*

loss+=�Ъ�       �	Zgv�ec�A�*

loss��;d˄       �	�w�ec�A�*

lossC��<̷�5       �	G�w�ec�A�*

loss�=^�M�       �	Qx�ec�A�*

loss 3�<s�ϩ       �	u�x�ec�A�*

loss�<9�       �	a�y�ec�A�*

lossv�<Z-ħ       �	�$z�ec�A�*

loss��<<�l0       �	Թz�ec�A�*

lossF�;�^�       �	OX{�ec�A�*

loss��<o�       �	B�{�ec�A�*

loss��;��\i       �	7�|�ec�A�*

loss n�<�Ɯ7       �	�&}�ec�A�*

lossW�:;7s��       �	��}�ec�A�*

lossV��<V���       �	�_~�ec�A�*

lossmL�;1:��       �	�ec�A�*

loss8<��4�       �	��ec�A�*

loss@d�<:gHl       �	F��ec�A�*

loss�s�=@O~y       �	���ec�A�*

loss��}<#�b�       �	阁�ec�A�*

loss�]<
u0_       �	�<��ec�A�*

loss�4�<�X�n       �	����ec�A�*

lossnv;�wH�       �	����ec�A�*

loss.=Y�k�       �	�+��ec�A�*

loss���=���       �	؄�ec�A�*

loss�;Zgȵ       �	�|��ec�A�*

loss}h�:��KD       �	�(��ec�A�*

loss�{�;;��       �	6Ȇ�ec�A�*

lossz}�<��w       �	�q��ec�A�*

loss�ܜ;���a       �	r��ec�A�*

loss�GD=z�~X       �	1��ec�A�*

lossV[n<���       �	���ec�A�*

loss11�;���       �	���ec�A�*

loss_�<���       �	����ec�A�*

loss��:(�qq       �	fK��ec�A�*

loss\$�:���       �	���ec�A�*

loss �<|�T�       �	��ec�A�*

loss�;<�n�       �	�'��ec�A�*

lossAƇ<J��       �	"ō�ec�A�*

loss�"=p���       �	�[��ec�A�*

lossi
+=��       �	���ec�A�*

loss.�q=>��s       �	����ec�A�*

lossEQ;���u       �	[%��ec�A�*

lossI��<h~e�       �	mǐ�ec�A�*

loss$/�:�V$�       �	0e��ec�A�*

loss A�<	�       �	i ��ec�A�*

loss��:`)��       �	k���ec�A�*

loss�~<;V�k       �	:=��ec�A�*

loss�6�=�\^0       �	wۓ�ec�A�*

loss�w(<IaS�       �	�s��ec�A�*

loss�L�<��E�       �	���ec�A�*

losss�~;�A��       �	f���ec�A�*

loss-5<:Ce        �	�;��ec�A�*

lossV|�;"�Ib       �	����ec�A�*

loss�{2<L��       �	�,��ec�A�*

loss��<i�X       �	�ͭ�ec�A�*

lossj�c=�3}       �	[_��ec�A�*

lossv�B<�)�$       �	R��ec�A�*

lossf�=6�	       �	c���ec�A�*

loss��<�[b�       �	���ec�A�*

loss��<tRRw       �	���ec�A�*

loss��= ��       �	����ec�A�*

loss%>�<�{V       �	�6��ec�A�*

loss
�]<�)�       �	ײ�ec�A�*

loss�֘;��+�       �	�k��ec�A�*

loss��7<��X       �	���ec�A�*

lossl&�<�V�       �	����ec�A�*

loss|h�;�`:       �	0I��ec�A�*

lossj��<o�Ff       �	3��ec�A�*

loss�-�;��E�       �	}v��ec�A�*

lossW�7;�*��       �	]p��ec�A�*

lossH>&=�`y       �	�3��ec�A�*

loss�L�< F��       �	LǸ�ec�A�*

loss�3<O!��       �	�[��ec�A�*

loss6	�=���       �	����ec�A�*

loss=��:��"�       �	����ec�A�*

lossnZh=h_o�       �	c*��ec�A�*

loss��<b��n       �	dʻ�ec�A�*

lossHc<�m�*       �	%]��ec�A�*

loss�n<��       �	j���ec�A�*

loss,��::��<       �	�߾�ec�A�*

loss�s;=�6�       �	�t��ec�A�*

lossi&�<�ԁ�       �	��ec�A�*

loss��/<��kA       �	���ec�A�*

lossŭ<�ύ�       �	�N��ec�A�*

lossst6=�id�       �	����ec�A�*

loss�y<���       �	���ec�A�*

loss\�!<�x�       �	rk��ec�A�*

loss���=��s       �	5
��ec�A�*

loss��;]��       �	Ҩ��ec�A�*

loss�QW<��       �	�D��ec�A�*

loss�O`=�G��       �	r���ec�A�*

loss���<]%:       �	0���ec�A�*

loss��F<�lz<       �	�=��ec�A�*

loss�<�<�       �	m���ec�A�*

loss$�t;GdH�       �	����ec�A�*

loss�U[<tÑ^       �	�5��ec�A�*

lossdd5;��       �	V���ec�A�*

lossjN={�       �	���ec�A�*

loss��=�	BH       �	+L��ec�A�*

lossB*<q�<       �	X���ec�A�*

lossN�;,�	       �	 ��ec�A�*

loss�u�;嵤�       �	`���ec�A�*

loss�m�;���	       �	'���ec�A�*

loss\�8=�ֈ�       �	�#��ec�A�*

lossG�:1�̤       �	����ec�A�*

loss2߱=�<x       �	��ec�A�*

loss���<��%4       �	����ec�A�*

loss\:��jv       �	�C��ec�A�*

loss��;���       �	���ec�A�*

lossc�4:9��       �	����ec�A�*

lossv�;<�1�       �	���ec�A�*

loss�=��9       �	v���ec�A�*

losst��=��       �	J]��ec�A�*

lossO�f9%�8�       �	����ec�A�*

loss�h�:*F�b       �	����ec�A�*

loss���<�f5<       �	�@��ec�A�*

lossEpC<�3��       �	����ec�A�*

loss��`=���8       �	nn��ec�A�*

loss�<!u�5       �	t��ec�A�*

loss�'];���~       �	^���ec�A�*

lossP��<��&       �	�/��ec�A�*

loss��=�6>       �	X���ec�A�*

lossl|y<m���       �	x_��ec�A�*

loss���<�$�}       �	g���ec�A�*

lossk[=S:�)       �	����ec�A�*

losse�&=y�}       �	���ec�A�*

loss$
�<��j�       �	���ec�A�*

lossmP=�|�       �	�G��ec�A�*

loss$W�;B$�       �	���ec�A�*

lossRj=?�       �	6r��ec�A�*

loss�E�<�fe�       �	B��ec�A�*

loss �;=�]>y       �	E���ec�A�*

loss��><����       �	28��ec�A�*

loss��b<W��       �	&S��ec�A�*

loss�\<3%�       �	����ec�A�*

loss#��:cU       �	����ec�A�*

loss�;bJh       �	!��ec�A�*

loss_�<Vh��       �	^���ec�A�*

lossRF�<//P       �	
���ec�A�*

loss�=I�       �	��ec�A�*

lossRx�;fa�       �	F���ec�A�*

loss�Fu<��8        �	�V��ec�A�*

lossؿ;<�Ó       �	����ec�A�*

loss~�<��=�       �	����ec�A�*

lossVK�;H�w       �	r��ec�A�*

loss���;�;�$       �	
��ec�A�*

loss,%�<BGp�       �	����ec�A�*

loss��:lp��       �	T��ec�A�*

loss�t{<��       �	����ec�A�*

lossX$"<\B0       �	l>��ec�A�*

loss�9�=�H�       �	_b��ec�A�*

loss#�<c�MR       �	���ec�A�*

loss`�<0c�C       �	����ec�A�*

loss��w:�0�       �	�=��ec�A�*

loss�!�=ر��       �	=���ec�A�*

loss�#(<���       �	�n��ec�A�*

lossR�:ԝ�       �	*��ec�A�*

loss��=����       �	����ec�A�*

loss≔;Vbz8       �	wL��ec�A�*

lossl$,<<s�       �	���ec�A�*

loss�Ls=�E�       �	�_��ec�A�*

loss���<:�H       �	.���ec�A�*

loss�ۄ= >$�       �	����ec�A�*

loss�V�;U-       �	�J��ec�A�*

lossK�=]��       �	����ec�A�*

lossƊ"=�|��       �	����ec�A�*

loss��:�0'�       �	c+��ec�A�*

loss|�<��`�       �	����ec�A�*

loss���<U[��       �	b��ec�A�*

lossJ>q<�1�y       �	����ec�A�*

loss�d�<f�y�       �	���ec�A�*

loss��0<H��       �	�)��ec�A�*

loss|�G<?M�1       �	[���ec�A�*

lossD�<��;U       �	�v��ec�A�*

loss��<����       �	�  fc�A�*

loss��6=��Ӝ       �	]�  fc�A�*

loss1�:3)Vo       �	
J fc�A�*

loss�tF;�W7\       �	9� fc�A�*

loss��<3T��       �	� fc�A�*

lossQ��<B�-7       �	1 fc�A�*

loss�	+<�i��       �	�� fc�A�*

lossoX=c��       �	$b fc�A�*

loss�<�Lc�       �	L� fc�A�*

loss��=�^��       �	B� fc�A�*

loss֌k<�=��       �	�/ fc�A�*

lossڰD<��c       �	i� fc�A�*

loss�Dx=hX�$       �	Eg fc�A�*

lossI��<z(�k       �	�? fc�A�*

loss�c�:�0tx       �	1#	 fc�A�*

lossCt<?x@       �	�	 fc�A�*

loss��;�c��       �	sg
 fc�A�*

lossO.<In��       �	/ fc�A�*

loss�;9��d       �	k� fc�A�*

losss�< a��       �	�� fc�A�*

loss�<}.�?       �	E fc�A�*

losss�;���c       �	rp fc�A�*

loss!��<��'\       �	� fc�A�*

lossd�<�el       �	ظ fc�A�*

loss�V�;��       �	�Y fc�A�*

losso�;ZS�T       �	�� fc�A�*

loss��z<7�%�       �	o� fc�A�*

lossߩ�;�b �       �	�B fc�A�*

loss3;�O#�       �	M� fc�A�*

loss�w�<ʄ�       �	B fc�A�*

loss1JA<��O�       �	P� fc�A�*

losso};�?��       �	�� fc�A�*

loss6]<�T~       �	'. fc�A�*

loss"> <�!��       �	s fc�A�*

loss��;�p�f       �	� fc�A�*

loss��>=����       �	C fc�A�*

lossX<n�)       �	I� fc�A�*

loss
!=���       �	ap fc�A�*

losst�<�t��       �	� fc�A�*

loss�{�;-3İ       �	s� fc�A�*

lossִK=���       �	O fc�A�*

loss��:�O��       �	�� fc�A�*

lossL��;(b�       �	�x fc�A�*

loss1H�<���N       �	A fc�A�*

loss��:wc<(       �	n� fc�A�*

loss,'�<uLff       �	�= fc�A�*

losst[={��       �	_� fc�A�*

loss�n!;��&l       �	�f fc�A�*

loss���<-5��       �	7� fc�A�*

loss췾;l�       �	��  fc�A�*

loss�;���M       �	;! fc�A�*

lossz;H:<#�-       �	��! fc�A�*

losso��9h��       �	�s" fc�A�*

loss��:[�ͼ       �	V# fc�A�*

loss��=	X�       �	��# fc�A�*

loss��=J��Q       �	=H$ fc�A�*

loss�7=5���       �	��$ fc�A�*

loss��;��b       �	�% fc�A�*

lossnW�9�h�/       �	I.& fc�A�*

lossW�(=]CQ.       �	��& fc�A�*

lossͰ;�       �	�t' fc�A�*

loss�ު;V�KD       �	�F( fc�A�*

lossD�<0]�       �	�( fc�A�*

loss=�+;���       �	܄) fc�A�*

loss���;���       �	H* fc�A�*

lossk=��7�       �	��* fc�A�*

loss��E<}A�       �	U+ fc�A�*

loss�[�<��:5       �	��+ fc�A�*

lossR�F=i�B       �	�, fc�A�*

lossfö<�8��       �	�!- fc�A�*

loss��7;ʴ7�       �	5�- fc�A�*

loss�Zj<I�       �	M. fc�A�*

losse|&:��d�       �	E�. fc�A�*

loss��(<7�?y       �	��/ fc�A�*

loss�;�h�       �	�I0 fc�A�*

lossS�</��       �	��0 fc�A�*

lossv��<��       �	Bz1 fc�A�*

lossF�;�=�       �	�2 fc�A�*

loss�2�=|�z�       �	X�2 fc�A�*

loss��;P�       �	%<3 fc�A�*

loss-u�<�[��       �	�3 fc�A�*

loss�;S^�(       �	�b4 fc�A�*

loss$l�<��hu       �	#�4 fc�A�*

loss�4<{�5|       �	�5 fc�A�*

lossNCG<��Jt       �	p
7 fc�A�*

lossR�<Q��       �	w�7 fc�A�*

lossrI=cqd\       �	�I8 fc�A�*

loss��<Xsϥ       �	��8 fc�A�*

loss���<v��       �	^�9 fc�A�*

loss��s;H��       �	,: fc�A�*

loss��>:]�/�       �	��: fc�A�*

loss�/�;�)�       �	ao; fc�A�*

loss7��<ŵ��       �	@< fc�A�*

lossH�=Z�=&       �	��< fc�A�*

loss��i=�H�       �	Z= fc�A�*

lossYܙ<
(|�       �	'�= fc�A�*

lossq�=N��D       �	K�> fc�A�*

loss���<�1Ș       �	�+? fc�A�*

loss%�r=&�#       �	��? fc�A�*

loss?K<ZK�       �	�T@ fc�A�*

lossfB?<`UH�       �	��@ fc�A�*

loss=�<"֕N       �	
�A fc�A�*

loss��;�xs�       �	TB fc�A�*

loss�'m<����       �	ĳB fc�A�*

loss/q�<H�i       �	�GC fc�A�*

loss�
<W���       �	s�C fc�A�*

loss@{�;ҁx�       �	7pD fc�A�*

loss:�~=� ]       �	rE fc�A�*

loss��;h$h\       �	�E fc�A�*

loss,�X=��p       �	�CF fc�A�*

loss ��<1���       �	j�F fc�A�*

loss �4<����       �	mqG fc�A�*

loss׎<�v�       �	�GH fc�A�*

loss qW<��ő       �	�H fc�A�*

loss�'=���T       �	C�I fc�A�*

loss�-<���       �	-J fc�A�*

loss��a=(Z��       �	�J fc�A�*

lossV$�;��9       �	5�K fc�A�*

lossq�%=�Ll�       �	]SL fc�A�*

loss�<,��m       �	)�L fc�A�*

loss]��=!��o       �	�M fc�A�*

loss�M=D<9       �	�-N fc�A�*

loss�Y;=ل�h       �	%�N fc�A�*

loss�(C=>�vT       �	DiO fc�A�*

loss8j`<>{�       �	�P fc�A�*

loss��P<���       �	w�P fc�A�*

loss�/=KJ�C       �	:Q fc�A�*

lossO�
=�T�       �	��R fc�A�*

loss�i=� ��       �	?qS fc�A�*

loss�H=(�(       �	&T fc�A�*

loss.\�<a�|�       �	��T fc�A�*

lossL2�;���       �	�^U fc�A�*

lossxu�;bP]       �	��U fc�A�*

loss,�Q<�?V       �	)�V fc�A�*

loss�1<�O-�       �	�3W fc�A�*

loss:w�<��s       �	��W fc�A�*

loss$�=�;�H       �	��X fc�A�*

lossVB~=��cC       �	�Y fc�A�*

loss�7m=��M       �	�"Z fc�A�*

loss��;p��       �	߿Z fc�A�*

loss`��<_�8       �	g[ fc�A�*

loss�<��8|       �	�\ fc�A�*

loss��<�ߛ�       �	ø\ fc�A�*

loss���;����       �	�] fc�A�*

loss��A=f�       �	�!^ fc�A�*

loss�&g<���@       �	b�^ fc�A�*

loss�/�<��       �	�R_ fc�A�*

loss@-a<z� .       �	��_ fc�A�*

loss��><{׀n       �	��` fc�A�*

loss���;ℕ       �	.a fc�A�*

lossoݿ;s#��       �	5�a fc�A�*

loss���;�BT_       �	Pb fc�A�*

lossw��<���       �	2�b fc�A�*

loss�$=���       �	�zc fc�A�*

loss�=J��N       �	d fc�A�*

loss��;U��       �	Ϊd fc�A�*

loss�=ަ!�       �	Ge fc�A�*

losstj�=��       �	��e fc�A�*

loss�.:#a�        �	gg fc�A�*

lossA-�;��Y�       �	�h fc�A�*

lossF�;�j_       �	�h fc�A�*

loss�6<���       �	�`i fc�A�*

lossA��<@�q       �	�j fc�A�*

loss��=���*       �	��j fc�A�*

loss�N=�,�)       �	�[k fc�A�*

loss֥);d�       �	�l fc�A�*

loss�;�Ç       �	Pm fc�A�*

loss��Y:C�'       �	 �m fc�A�*

loss�A�:�U�       �	j�n fc�A�*

loss
��;���       �	�2o fc�A�*

loss�5<��       �	k�o fc�A�*

loss���;܈H;       �	Z�p fc�A�*

lossv,�;��ʆ       �	2sq fc�A�*

loss��P<����       �	!�r fc�A�*

lossl��:ah�c       �	q�s fc�A�*

lossÇ�=km�{       �	�jt fc�A�*

loss�~�<�6	�       �	u fc�A�*

loss�JY<,4��       �	 �u fc�A�*

lossu8=K�Y       �	=Ev fc�A�*

lossc�;�S��       �	��v fc�A�*

lossj�C;sU+       �	�w fc�A�*

loss�\=��A       �	qx fc�A�*

lossA�F<դ��       �	w�x fc�A�*

loss�э=w�[�       �	Xy fc�A�*

loss.��;:C!       �	��y fc�A�*

loss#
p=�-ؖ       �	��z fc�A�*

loss���;p���       �	 <{ fc�A�*

loss�;jE�       �	��{ fc�A�*

loss�F�;�^W�       �	Bz| fc�A�*

lossEo)=9]<�       �	z} fc�A�*

loss�%�;<ȏ4       �	N�} fc�A�*

loss�[�<��A�       �	J~ fc�A�*

loss �<`���       �	��~ fc�A�*

loss�K)=�)��       �	=| fc�A�*

loss���;%q3�       �	�� fc�A�*

loss
mj<�ܝ       �	܀ fc�A�*

lossJ=�RO       �	�v� fc�A�*

loss�@�<��S       �	�� fc�A�*

loss�{<,P �       �	R�� fc�A�*

loss�	�=]?�       �	�[� fc�A�*

lossT�K=C�Q�       �	��� fc�A�*

loss�j�<�       �	��� fc�A�*

loss��<�~c)       �	�L� fc�A�*

loss���<���       �	�� fc�A�*

losst�*<&�r       �	؁� fc�A�*

lossb<(V�       �	u� fc�A�*

loss���;{;;�       �	0�� fc�A�*

loss �<�/�       �	�� fc�A�*

lossp�<h�       �	 � fc�A�*

loss�B1=�5       �	�� fc�A�*

loss)Vs;G�r�       �	TT� fc�A�*

loss�<��a�       �	��� fc�A�*

loss�s<Tlv�       �	��� fc�A�*

loss��Q;���f       �	㈌ fc�A�*

losswf�<����       �	�r� fc�A�*

loss���;.F�k       �	k� fc�A�*

loss��<�V       �	�� fc�A�*

loss��;Z�t�       �	R,� fc�A�*

lossj�:d�d       �	}͐ fc�A�*

lossib�;?q2�       �	Ho� fc�A�*

loss ��;.��       �	� fc�A�*

loss�vM=�?��       �	Ҭ� fc�A�*

loss�=,�^       �	^J� fc�A�*

loss���<.i�       �	�� fc�A�*

lossG<�Fl�       �	�� fc�A�*

loss�}=d�       �	"� fc�A�*

loss��<,Z��       �	伕 fc�A�*

loss�[�=��       �	
f� fc�A�*

lossit";U7�       �	� fc�A�*

loss���<ڕ�       �	��� fc�A�*

lossj:`=Gf�       �	�J� fc�A�*

loss]Y�<]K�       �	�� fc�A�*

loss��;-i�7       �	��� fc�A�*

loss.@h<ܧ�       �	Y3� fc�A�*

lossfR:d���       �	{ݚ fc�A�*

loss���;��y       �	ˆ� fc�A�*

loss¾;�p8�       �	�*� fc�A�*

loss_(=Sz�       �	М fc�A�*

loss|\=}���       �	�j� fc�A�*

loss�F6<o��       �	J	� fc�A�*

lossd��;�>T       �	u�� fc�A�*

loss��
>�돧       �	�H� fc�A�*

loss�U�<x�g�       �	�� fc�A�*

lossd�.=ѮF/       �	��� fc�A�*

loss��$=k�       �	� � fc�A�*

loss-�;(%��       �	��� fc�A�*

lossq �<x��W       �	�Z� fc�A�*

lossL��<Ѩ�       �	�� fc�A�*

loss�̤<�'�       �	��� fc�A�*

loss��3<����       �	2V� fc�A�*

loss��U<�֥%       �	�� fc�A�*

loss��<�/��       �	^�� fc�A�*

lossCd;_6�g       �	�6� fc�A�*

loss(A;�	Q       �	�ئ fc�A�*

losse�<��       �	|� fc�A�*

loss		=���       �	� fc�A�*

loss��=h���       �	 fc�A�*

lossn�2=�B)H       �	TT� fc�A�*

loss�O<�:       �	�� fc�A�*

loss��m<!P��       �	뎪 fc�A�*

loss�C�=vIp0       �	 8� fc�A�*

loss��;�H?8       �	�ӫ fc�A�*

loss�C(<cs\       �	�l� fc�A�*

loss�%=>�       �	�� fc�A�*

lossjg@<W�f�       �	��� fc�A�*

loss��<����       �	�e� fc�A�*

loss�8R=U�TK       �	��� fc�A�*

loss�?�<j�a1       �	��� fc�A�*

lossa��;X��       �	�H� fc�A�*

lossi��;wе       �	�� fc�A�*

lossҐ<�?��       �	� fc�A�*

losss��<����       �	>!� fc�A�*

loss�+�<�E�       �	Z-� fc�A�*

loss�[J=8k Z       �	�%� fc�A�*

lossؤ/<�/�$       �	ε fc�A�*

loss?j�:2o�       �	w� fc�A�*

lossN�];��	�       �	v� fc�A�*

loss4B=N6�       �	�ŷ fc�A�*

loss�<�01�       �	Qk� fc�A�*

loss1��<��,       �	�
� fc�A�*

loss��@<�>�       �	�� fc�A�*

loss��k<x)r�       �	iT� fc�A�*

lossq��<6@Z�       �	�� fc�A�*

loss�@;�C��       �	⑻ fc�A�*

loss��k<��C       �	�0� fc�A�*

loss���;/�}�       �	Fͼ fc�A�*

lossq ]<�N>�       �	z� fc�A�*

loss{�;���       �	� fc�A�*

loss�Q=��aa       �	C�� fc�A�*

loss��=�l�       �	�J� fc�A�*

loss[�Z;�g�]       �	�� fc�A�*

loss�G<��\Y       �	i�� fc�A�*

loss�,=p�:�       �	�%� fc�A�*

loss2 �={؈%       �	�� fc�A�*

loss*W:<!���       �	O\� fc�A�*

loss�G;A�A       �	��� fc�A�*

loss�68=�+B       �	��� fc�A�*

loss`�J<�� �       �	$)� fc�A�*

loss��<��G�       �	;�� fc�A�*

loss<k�<�8�=       �	�[� fc�A�*

loss�9�:A#nI       �	�7� fc�A�*

loss�"2=���       �	�� fc�A�*

loss2&�<�/m       �	Kv� fc�A�*

loss#�\<Q���       �	�� fc�A�*

loss�!X<ޙM       �	�� fc�A�*

loss.6L<��tH       �	p� fc�A�*

loss�#/<xP�j       �	�� fc�A�*

loss��;5��       �	��� fc�A�*

loss��&<>��k       �	�<� fc�A�*

loss�0<&]�       �	��� fc�A�*

loss���;ѩo�       �	�w� fc�A�*

loss�@�<?��       �	�� fc�A�*

loss7�<�0�       �	*�� fc�A�*

losso�=(�~"       �	l�� fc�A�*

lossv�<;���       �	Z-� fc�A�*

loss�L�<�v�N       �	��� fc�A�*

lossHN;h���       �	��� fc�A�*

loss��<��Æ       �	.7� fc�A�*

loss���<l�v       �	��� fc�A�*

lossJ��=�z�       �	��� fc�A�*

loss�cE=���       �	�[� fc�A�*

lossCC�<��&       �	��� fc�A�*

lossQ�:zȾ�       �	2�� fc�A�*

loss_@V=Ճn       �	�N� fc�A�*

loss�=.�wy       �	��� fc�A�*

loss=r��'       �	��� fc�A�*

lossLI)=୳       �	E)� fc�A�*

loss8�<�@�       �	��� fc�A�*

loss�:�<1Arq       �	�S� fc�A�*

loss�g�<��"�       �	��� fc�A�*

loss�<,<��~w       �	G � fc�A�*

loss�4�;�S�N       �	ȴ� fc�A�*

loss6�<?�'�       �	�R� fc�A�*

lossO��;� �D       �	S�� fc�A�*

loss�Mo<��L�       �	s�� fc�A�*

loss���<nR��       �	�� fc�A�*

lossqm<Nj]       �	��� fc�A�*

loss-��;Kڵ�       �	�J� fc�A�*

loss0�;���
       �	��� fc�A�*

loss�##=O4ՠ       �	��� fc�A�*

loss	7-;My?�       �	�"� fc�A�*

loss[�>=߃�;       �	��� fc�A�*

loss@T<FW�       �	b� fc�A�*

loss��<�       �	k� fc�A�*

losst>(;#z�M       �	��� fc�A�*

lossE��;R��       �	hA� fc�A�*

loss\�<�?       �	��� fc�A�*

loss�~u;N� s       �	w� fc�A�*

lossZ�e=>���       �	x� fc�A�*

lossz+�:�Ӭa       �	��� fc�A�*

loss철<�0nu       �	 :� fc�A�*

lossW�
>����       �	�� fc�A�*

loss
#<���R       �	b� fc�A�*

loss�'�;VA�       �	��� fc�A�*

loss���;�l       �	ȕ� fc�A�*

loss� �;�N�J       �	I*� fc�A�*

loss&�Z;ѿ��       �	��� fc�A�*

loss�c�=�R�w       �	E�� fc�A�*

loss��;6��	       �	~Q� fc�A�*

loss��=<ε8b       �	��� fc�A�*

loss=GH<�7�=       �	�� fc�A�*

lossxy<�A6       �	Ts� fc�A�*

lossG=,�|       �	�� fc�A�*

loss �<?��       �	C�� fc�A�*

loss�W�:K���       �	�G� fc�A�*

loss}�<�^zT       �	��� fc�A�*

lossfL#;���       �	�y� fc�A�*

loss�9C;��>e       �	{� fc�A�*

lossx�v<��%h       �	K�� fc�A�*

loss-W�;f��\       �	�� fc�A�*

loss��@<n��       �	��� fc�A�*

lossc�?=��~       �	�� fc�A�*

loss��%<���g       �	��� fc�A�*

loss�,<O��       �	�U� fc�A� *

loss�u�<�R�       �	ܝ� fc�A� *

lossM�;�1(       �	z4� fc�A� *

loss��$=����       �	��� fc�A� *

loss�q<�^�       �	s� fc�A� *

loss�y<�c�       �	�$� fc�A� *

lossK�:�K�8       �	��� fc�A� *

loss];I<3��       �	Ed� fc�A� *

lossw�,=�9�       �	� fc�A� *

loss*6[;˘�*       �	��� fc�A� *

loss��G=�Q��       �	�\� fc�A� *

lossNM<J\��       �	��� fc�A� *

loss���;���       �	�� fc�A� *

loss���<+]^�       �	�F� fc�A� *

loss���;)�?�       �	!�� fc�A� *

lossl�:UjR       �	m�� fc�A� *

loss��<w�Q�       �	f� fc�A� *

loss��J;�]�       �	�fc�A� *

loss�m�<ٚ^�       �	��fc�A� *

loss���<0��$       �	bfc�A� *

loss��G<��F�       �	��fc�A� *

loss�{�=��^       �	\�fc�A� *

loss�n<_�9        �	�&fc�A� *

loss!�: ��g       �	$Cfc�A� *

loss$W�<���       �	L�fc�A� *

loss��:��RP       �	��fc�A� *

loss�/;W�(b       �	�fc�A� *

lossF1:4K��       �	^.fc�A� *

loss�+=u�       �	��fc�A� *

loss�+<�^�I       �	�	fc�A� *

loss%h�;�k%       �	6
fc�A� *

loss�E�<��I�       �	��
fc�A� *

loss��;`�\�       �	�efc�A� *

loss(�4=O�p       �	>fc�A� *

loss��b=���       �	��fc�A� *

loss@�;��Sn       �	�;fc�A� *

loss�K�<��!       �	��fc�A� *

loss�q�:���       �	�xfc�A� *

loss��l:�,�P       �	K fc�A� *

loss=��;t���       �	��fc�A� *

loss�� ;\ɐ�       �	�Sfc�A� *

lossT]�;��       �	O�fc�A� *

lossd�W;���       �	�fc�A� *

loss6�$;o��       �	)!fc�A� *

lossk�<[���       �	��fc�A� *

loss�½< 3s�       �	[^fc�A� *

loss���92��       �	]�fc�A� *

loss�%:����       �	h�fc�A� *

loss�5:B6�f       �	a4fc�A� *

loss|��;`�M       �	��fc�A� *

lossX<r<��       �	jfc�A� *

loss��49�`0&       �	�fc�A� *

lossW�2;�B�N       �	¢fc�A� *

loss�H>S��       �	
Hfc�A� *

loss�9���       �	��fc�A� *

loss3�=�L�       �	��fc�A� *

loss���< ���       �	�&fc�A� *

losshG}<�|�       �	:�fc�A� *

lossT|�;+�
       �	*pfc�A� *

lossp�<�f�       �	�fc�A� *

lossU�=
��F       �	ȷfc�A� *

lossjP<d�       �	�Tfc�A� *

loss͒�;簱       �	�fc�A� *

loss'l<j��F       �	��fc�A� *

loss��$;z1j�       �	6>fc�A� *

loss
�y=U	�        �	��fc�A� *

lossr��<��       �	F{ fc�A� *

lossq�<]��       �	�!fc�A� *

loss��=z)�)       �	��!fc�A� *

lossӦ=����       �	�f"fc�A� *

loss�(�;��)       �	�#fc�A� *

loss��<7D       �	4�#fc�A� *

lossm+�<��H�       �	`?$fc�A� *

loss�?<3���       �	��$fc�A� *

loss,��;VߑM       �	�u%fc�A� *

loss<=b�3       �	�&fc�A� *

loss���;�	i       �	��&fc�A� *

loss���:�`�       �	�A'fc�A� *

loss�[<�l��       �	�'fc�A� *

loss�>;D��       �	o(fc�A� *

loss���;9�T�       �	y)fc�A� *

loss��]=]��       �	�)fc�A� *

loss���<�Aw       �	��*fc�A� *

lossS��<m�       �	*+fc�A� *

loss�;�4�3       �	�+fc�A� *

lossh <*��j       �	a,fc�A� *

lossŕ<;��       �	��,fc�A� *

loss��;o�=       �	q�-fc�A� *

lossV =����       �	+.fc�A� *

loss�&<P�r       �	��.fc�A� *

loss ��;��t&       �	1\/fc�A� *

loss�[�<�Hܜ       �	eT0fc�A� *

loss��=�&~c       �	p	1fc�A� *

loss/<)��       �	��1fc�A� *

lossA��:���       �	TS2fc�A� *

loss="O;\��       �	�83fc�A� *

loss<U<�g&       �	/4fc�A� *

lossŨ><e�=�       �	}�4fc�A� *

lossU�<\ \�       �	�u5fc�A� *

loss{٘;B���       �	L66fc�A� *

lossc�=.ʍs       �	s7fc�A� *

loss�4f=#��       �	��7fc�A� *

loss1�<��       �	cd8fc�A� *

loss6o�:���       �	o9fc�A� *

loss�E�;���?       �	��9fc�A� *

loss��;Q��*       �	�jQfc�A� *

loss���;�ݎ�       �	��Qfc�A� *

loss���;++u�       �	��Rfc�A� *

lossl?�<�F�       �	�<Sfc�A� *

loss��<z�       �	��Sfc�A� *

losski<��!)       �	�|Tfc�A� *

lossMw7<$�q�       �	rUfc�A� *

loss��N<��i       �	��Ufc�A� *

loss��u=�#�       �	�AVfc�A� *

lossO>	=�}�       �	 �Vfc�A� *

lossW�;��M|       �	�Wfc�A� *

loss
5}=Uܫ'       �	4Xfc�A� *

loss�:E;aK �       �	�Xfc�A� *

loss���<t2��       �	��Yfc�A� *

loss%�A<&E�       �	x)Zfc�A� *

loss%�=Ō��       �	��Zfc�A� *

loss|%;M��U       �	1{[fc�A� *

lossT�;#�S<       �	�\fc�A� *

loss4�"<iN�       �	X�\fc�A� *

loss�_�=}��       �	�=]fc�A� *

loss}�P<��F�       �	�]fc�A� *

lossCou=j:�T       �	��^fc�A� *

lossDc�<fk~�       �	�/_fc�A� *

losst�q=�ŕ�       �	K�_fc�A�!*

loss��<�Mد       �	�c`fc�A�!*

loss_=�<(���       �	{�`fc�A�!*

loss
=4<�}��       �	�afc�A�!*

loss:�;�7GG       �	Ebfc�A�!*

loss��<�ً       �	
cfc�A�!*

loss���;�_E�       �	�cfc�A�!*

lossɒ<.���       �	J�dfc�A�!*

loss�f4;!5y�       �	�;efc�A�!*

loss:y�;��P�       �	��efc�A�!*

loss�M�<J�gM       �	�kffc�A�!*

loss��-;�@��       �	�gfc�A�!*

loss_f$<"��       �	��gfc�A�!*

lossJ�;9j=       �	�*hfc�A�!*

loss�	=4��       �	�hfc�A�!*

loss
�\=G��       �	�Yifc�A�!*

loss���<��a       �	N�ifc�A�!*

loss8�8=^;�[       �	��jfc�A�!*

loss?p<R�֒       �	�Skfc�A�!*

lossl��:b��%       �	��kfc�A�!*

loss�	_<���u       �	Bmfc�A�!*

loss!��;��MW       �	��mfc�A�!*

loss��y=�@��       �	�nfc�A�!*

lossS�<<�vͶ       �	vofc�A�!*

loss$P�;��{       �	V�ofc�A�!*

lossW}*;���       �		Opfc�A�!*

loss��9q�"}       �	��pfc�A�!*

loss�S
<�^
!       �	��qfc�A�!*

loss���<�p��       �	�rfc�A�!*

loss���<����       �	�Wsfc�A�!*

loss*O=Yh�       �	�btfc�A�!*

loss�V�=2��       �	Z�tfc�A�!*

loss9[:VjH�       �	�ufc�A�!*

loss�_;�X.�       �	3Pvfc�A�!*

loss⁖:�뮃       �	��wfc�A�!*

loss�F�<���       �	4Jxfc�A�!*

loss�,�<b4�       �	&�yfc�A�!*

lossf��<����       �	U�zfc�A�!*

lossᠼ<��       �	�J{fc�A�!*

loss��;<��       �	;|fc�A�!*

loss�K�<�s��       �	�L}fc�A�!*

lossґ<;�2+       �	�}fc�A�!*

loss{յ:�h�       �	ɓ~fc�A�!*

loss�;q{��       �	UPfc�A�!*

loss�\�;p�,       �	qZ�fc�A�!*

loss|@<I�       �	�w�fc�A�!*

loss�c=5*\D       �	�"�fc�A�!*

lossE�[;��{       �	�΂fc�A�!*

loss;Q;>��2       �	��fc�A�!*

lossj�=�J��       �	X��fc�A�!*

loss�q;��	�       �	�Q�fc�A�!*

lossӴ;���       �	��fc�A�!*

loss��7;>��*       �	���fc�A�!*

loss��:���T       �	 D�fc�A�!*

lossֆ;��d�       �	!�fc�A�!*

loss���<4;��       �	���fc�A�!*

losss��<މ��       �	�?�fc�A�!*

loss.<V&�m       �	��fc�A�!*

loss.=�ƹ�       �	'��fc�A�!*

loss�	><��       �	(*�fc�A�!*

loss<��<q~M�       �	�ϋfc�A�!*

loss�13;�1;       �	,}�fc�A�!*

loss}��;A�~�       �	�,�fc�A�!*

loss<�f<=���       �	��fc�A�!*

lossn��=�P�       �	���fc�A�!*

loss�M><��F�       �	�2�fc�A�!*

loss�%�</��       �	�܏fc�A�!*

loss��:<��_$       �	w��fc�A�!*

lossl7)<�X��       �	&�fc�A�!*

lossM"�:]�@]       �	�Ǒfc�A�!*

loss�G�;���       �	bj�fc�A�!*

loss�<8ׂ       �	�fc�A�!*

loss�&�:�C$�       �	.��fc�A�!*

loss��|=��       �	�H�fc�A�!*

loss1�Q=Or+       �	T�fc�A�!*

loss�F=�1|�       �	z�fc�A�!*

loss�A;��7[       �	�fc�A�!*

loss�!<@^́       �	Ზfc�A�!*

loss�)�:�%       �	��fc�A�!*

loss��:,R�       �	���fc�A�!*

lossv;���       �	c)�fc�A�!*

loss*�<�F��       �	l˙fc�A�!*

loss�(	=���       �	�d�fc�A�!*

loss�T4<}F�s       �	��fc�A�!*

loss�]R<nl߼       �	���fc�A�!*

loss o	= 2l       �	.=�fc�A�!*

lossed�;�S�       �	�؜fc�A�!*

lossaC=�`�1       �	�f�fc�A�!*

loss�;�_�W       �	/��fc�A�!*

loss� =Րt�       �	w��fc�A�!*

loss��:����       �	�P�fc�A�!*

lossF�3<q�%D       �	��fc�A�!*

loss�;���^       �	���fc�A�!*

lossMHT;&���       �	�'�fc�A�!*

loss\�<c��        �	���fc�A�!*

loss;�~<Єab       �	&W�fc�A�!*

loss�=����       �	��fc�A�!*

lossI�;���       �	b��fc�A�!*

loss.��=	�B�       �	��fc�A�!*

lossQ5;�k       �	Y��fc�A�!*

lossh�F=�窝       �	�b�fc�A�!*

loss�M<�Sw       �	l[�fc�A�!*

loss�e"=W��b       �	0�fc�A�!*

lossͤx;�|Z�       �	u��fc�A�!*

loss<&<��'       �	<N�fc�A�!*

loss�q1=�8n       �	!�fc�A�!*

loss
�Z=h[y�       �	��fc�A�!*

loss\�A<�T��       �	�/�fc�A�!*

lossx�5=��$�       �	�Ϊfc�A�!*

lossz�M<F�       �	fl�fc�A�!*

loss�)1=�b       �	c�fc�A�!*

lossR�d<�X��       �	���fc�A�!*

loss�7�;���       �	f2�fc�A�!*

loss�U�;�}�       �	}\�fc�A�!*

loss�j�;,N��       �	���fc�A�!*

loss1ȝ:@=�n       �	{��fc�A�!*

loss�&�:�>�       �	{I�fc�A�!*

loss�d<E�Mn       �	m�fc�A�!*

loss��X=�1L       �	1|�fc�A�!*

lossS�c<�       �	/�fc�A�!*

lossx��<6��       �	Ͽ�fc�A�!*

loss��0=��Y�       �	o�fc�A�!*

loss��;_�	�       �	��fc�A�!*

loss���<Lm�       �	��fc�A�!*

loss$~@;i���       �	'0�fc�A�!*

lossWĩ;L�	       �	Sʵfc�A�!*

loss�:3;��>P       �	�^�fc�A�!*

lossMX<$���       �	���fc�A�!*

loss��;=[ޟ�       �	~��fc�A�"*

lossk0=e���       �	3�fc�A�"*

lossx��<��	m       �	kظfc�A�"*

loss\�;YIO6       �	���fc�A�"*

loss9N�;+vѾ       �	��fc�A�"*

loss�	�;�X�       �	���fc�A�"*

loss�8�<���M       �	9c�fc�A�"*

loss�n�<�g��       �	3��fc�A�"*

losss
�<����       �	��fc�A�"*

loss���=]O�       �	�&�fc�A�"*

lossc�:�Ώ�       �	�ʽfc�A�"*

loss��<����       �	�d�fc�A�"*

loss���9K;ʖ       �	��fc�A�"*

loss� �<�o�       �	eſfc�A�"*

loss@Q;be�       �	@j�fc�A�"*

loss�>�:ȝ��       �	6�fc�A�"*

lossq��=�*       �	���fc�A�"*

loss��:��I�       �	�;�fc�A�"*

loss���:N,       �	���fc�A�"*

loss10'=�I�       �	�p�fc�A�"*

losslg�;�
�>       �	�fc�A�"*

loss�"�:~A�s       �	I��fc�A�"*

loss�!i<$�~=       �	,D�fc�A�"*

loss�-:��uj       �	w��fc�A�"*

loss#��;��ð       �	�}�fc�A�"*

loss��<��i       �	$�fc�A�"*

loss�ؖ<��0       �	���fc�A�"*

lossN��<��k        �	ni�fc�A�"*

lossc��<��w�       �		�fc�A�"*

loss`s:��f0       �	Z��fc�A�"*

loss_Na<�vS�       �	Gs�fc�A�"*

loss�
d;_��       �	R
�fc�A�"*

lossIo;K�gE       �	i��fc�A�"*

loss�~;
��H       �	^I�fc�A�"*

loss!�R<���       �	��fc�A�"*

lossLX�<���       �	@��fc�A�"*

loss2�=�Ԇ\       �	6�fc�A�"*

loss!��<�M       �	ؼ�fc�A�"*

loss`�:�;!�       �	}Y�fc�A�"*

lossR=A���       �	��fc�A�"*

loss3P�<�y��       �	��fc�A�"*

loss���;ל�=       �	�4�fc�A�"*

loss�3�;5r>       �	��fc�A�"*

loss���9R�'       �	��fc�A�"*

loss��G;nl�>       �	x��fc�A�"*

loss���<���       �	t��fc�A�"*

loss��A<HC��       �	��fc�A�"*

loss{�L<Wc+�       �	+�fc�A�"*

loss��|;�}!X       �	\��fc�A�"*

loss�3�;:�$       �	�m�fc�A�"*

loss��<�Z�       �	m�fc�A�"*

loss�5<H
�       �	w��fc�A�"*

lossv��<w�m�       �	9c�fc�A�"*

loss5�;��}�       �	$�fc�A�"*

loss�*U;U���       �	��fc�A�"*

loss��<'�0�       �	AJ�fc�A�"*

loss3��<����       �	���fc�A�"*

lossA��<����       �	w��fc�A�"*

loss�0<!M�       �	��fc�A�"*

loss�̿; ȍV       �	h��fc�A�"*

loss�Z9<���q       �	I�fc�A�"*

loss�y,;��c�       �	��fc�A�"*

loss!A�;��<B       �	���fc�A�"*

loss&�:=ϒ^�       �	�&�fc�A�"*

lossm��<7��       �	��fc�A�"*

loss�H~=��\       �	lZ�fc�A�"*

loss�]=���O       �		��fc�A�"*

loss��
=�j��       �	��fc�A�"*

lossV��=F�}6       �	Z-�fc�A�"*

loss��=\��       �	r��fc�A�"*

loss@��:�(+Y       �	f�fc�A�"*

loss� �<��       �	q��fc�A�"*

loss[�=A��       �	t��fc�A�"*

lossڻV;12Q�       �	|+�fc�A�"*

loss��x<�j�       �	<��fc�A�"*

lossf y<|�5�       �	�S�fc�A�"*

loss��<��e�       �	���fc�A�"*

loss6��;Y���       �	}�fc�A�"*

loss�N%=ѿ<�       �	��fc�A�"*

loss_��;Y��/       �	���fc�A�"*

loss���;�]0�       �	
H�fc�A�"*

loss�%�<����       �	��fc�A�"*

lossxD�<W��       �	�q�fc�A�"*

loss�g<����       �	�fc�A�"*

lossܱO<z��H       �	���fc�A�"*

loss2w�<�b�       �	�<�fc�A�"*

loss���<��$�       �	���fc�A�"*

loss6F<wZL�       �	�n�fc�A�"*

loss�8�;�P{r       �	��fc�A�"*

loss <��!       �	���fc�A�"*

lossT��<�rf7       �	>=�fc�A�"*

lossM�<�28�       �	���fc�A�"*

lossʡ�<7�       �	v��fc�A�"*

loss��o<�b��       �	�#�fc�A�"*

loss��<t�Q�       �	���fc�A�"*

lossB��;�^�       �	�_�fc�A�"*

lossn
�;�>C�       �	�#�fc�A�"*

loss֍s=jT��       �	
��fc�A�"*

loss���<ޛ�,       �	9|�fc�A�"*

loss�>���@       �	^�fc�A�"*

lossC1B<�q�       �	���fc�A�"*

loss3�<?ѡm       �	IG�fc�A�"*

loss��";M�\�       �	��fc�A�"*

lossL-+:%��       �	���fc�A�"*

loss|�=VY�       �	I�fc�A�"*

loss�[�<��       �	���fc�A�"*

loss7>z;le�       �	��fc�A�"*

loss��=͟�       �	�1�fc�A�"*

loss���;��       �	���fc�A�"*

loss�a=+�`�       �	�e�fc�A�"*

loss��!<δ�       �	���fc�A�"*

loss#�<�>R{       �	X��fc�A�"*

loss!��<I��       �	�,�fc�A�"*

loss�1�<��V       �	��fc�A�"*

loss̰L:C��       �	Lm fc�A�"*

loss��=���S       �	yfc�A�"*

loss��:=�(I�       �	ԛfc�A�"*

loss���<@4�       �	�5fc�A�"*

loss���<�r�       �	!�fc�A�"*

loss��<�0�P       �	hfc�A�"*

loss-�N<,|�       �	� fc�A�"*

loss��\=��m       �	�fc�A�"*

loss��(;X��]       �	</fc�A�"*

lossI�p=��.t       �	��fc�A�"*

loss�4�=���       �	Vdfc�A�"*

loss��<h��       �	fc�A�"*

lossRA�:j��       �	��fc�A�"*

lossC�w<�8�       �	�4fc�A�"*

losst�=8bOq       �	J�fc�A�#*

loss]/�<aܞ       �	�h	fc�A�#*

loss�W�<�hf�       �	�fc�A�#*

lossξ�<�6:x       �	0Jfc�A�#*

loss�/�=�o`       �	��fc�A�#*

lossFI�=��K;       �	��fc�A�#*

loss��<�V`       �	�fc�A�#*

loss;�%=�6F�       �	C�fc�A�#*

lossj|$=�t��       �	�;fc�A�#*

losso�=��#       �	��fc�A�#*

lossAT�:;$0       �	b�fc�A�#*

loss�/�9$�       �	�_fc�A�#*

lossh�:J�S        �	��fc�A�#*

losss��;�R�]       �	N�fc�A�#*

loss�>:?��       �	fc�A�#*

loss2jA<>
�L       �	�fc�A�#*

lossl��<t���       �	�Vfc�A�#*

loss�KD:{�V'       �	fc�A�#*

lossW�_=����       �	"�fc�A�#*

loss�=���)       �	�~fc�A�#*

loss�i=�~L�       �	�fc�A�#*

loss��
<��kM       �	M�fc�A�#*

loss�H0=Ҷ=O       �	�[fc�A�#*

lossl �;���B       �	 fc�A�#*

lossR�<A���       �	ԛfc�A�#*

loss�g�<�:`       �	K?fc�A�#*

loss�o�=t��       �	4�fc�A�#*

lossN5;�!�       �	1{fc�A�#*

lossF��<�       �	�<fc�A�#*

loss�><�v��       �	��fc�A�#*

lossl��;�`       �	!� fc�A�#*

loss�3�<�Y�L       �	�0!fc�A�#*

loss�c<�*       �	!�!fc�A�#*

loss�=t���       �	Qf"fc�A�#*

lossOYb<]��       �	��"fc�A�#*

loss
��;�ɋd       �	V�#fc�A�#*

loss۾�<�JL)       �	1$fc�A�#*

lossvR�;=��       �	��$fc�A�#*

loss\�E<��di       �	�f%fc�A�#*

loss�7<��TV       �	s�%fc�A�#*

lossz&<MӬ�       �	j�&fc�A�#*

loss�=x;-��       �	&'fc�A�#*

loss��`=��0�       �	9�'fc�A�#*

loss��:<p�;       �	�j(fc�A�#*

loss��;B�{       �	��(fc�A�#*

lossd��<>�"       �	^�)fc�A�#*

loss㹃<�z�_       �	�`*fc�A�#*

lossl��;�H�       �	
�*fc�A�#*

loss��=��HR       �	��+fc�A�#*

loss}�;�	W       �	?,fc�A�#*

loss	A�=�V�&       �	i�,fc�A�#*

loss��_<�0��       �	;-fc�A�#*

loss?�(<���       �	��-fc�A�#*

lossj��;w3��       �	��.fc�A�#*

loss���<Չ�       �	u>/fc�A�#*

loss���<�       �	�/fc�A�#*

loss�6�;4���       �	�i0fc�A�#*

loss��^<��j�       �	\1fc�A�#*

loss��;*p̫       �	��1fc�A�#*

loss-ޖ<���       �	C92fc�A�#*

loss��;cJ	       �	��2fc�A�#*

loss�<)Z�       �	��4fc�A�#*

lossO#�;:ڕ       �	�5fc�A�#*

loss#\�;PE��       �	=�5fc�A�#*

loss��w;U�B�       �	�M6fc�A�#*

lossi;=ԧ�@       �	?�6fc�A�#*

loss��<o( �       �	\t7fc�A�#*

loss�ǳ;B�V       �	�	8fc�A�#*

loss�S�<:�M<       �	�8fc�A�#*

lossv��<9pJ       �	�E9fc�A�#*

lossI�H=Y���       �	��:fc�A�#*

lossB�;'w\       �	�%;fc�A�#*

lossI��:>!��       �	g�;fc�A�#*

loss�|�<\��       �	�L<fc�A�#*

lossZ�=j��       �	v�<fc�A�#*

loss2f�<���[       �	��=fc�A�#*

loss��;t,Bv       �	II>fc�A�#*

losshY�;�>ں       �	��>fc�A�#*

loss���<�x@�       �	�r?fc�A�#*

loss$;���U       �	��?fc�A�#*

loss��<�6E       �	�@fc�A�#*

lossʱ�<0(��       �	f-Afc�A�#*

loss�u~;_�       �	��Afc�A�#*

loss�w<�kM       �	[aBfc�A�#*

lossv%=��RW       �	��Bfc�A�#*

loss�d<z� �       �	C�Cfc�A�#*

loss:��<J       �	�Dfc�A�#*

lossquo<��*�       �	��Dfc�A�#*

lossߎ�<�,^�       �	tFEfc�A�#*

lossT�h=��       �	w�Efc�A�#*

loss�<形�       �	"pFfc�A�#*

lossΏM<Zl�I       �	�Gfc�A�#*

loss%c�;�3�       �	¢Gfc�A�#*

lossD��;9�$n       �	�9Hfc�A�#*

lossʉn<�_�       �	_�Hfc�A�#*

lossӷ;=!W^       �	~pIfc�A�#*

loss <�yj       �	9Jfc�A�#*

loss��;���       �	"�Jfc�A�#*

loss>�=�:Jq       �	�>Kfc�A�#*

loss�Q,=Z���       �	o�Kfc�A�#*

loss�m<���       �	�kLfc�A�#*

loss���<U�7       �	EMfc�A�#*

loss�7�;��xi       �	��Mfc�A�#*

loss�oz=�s_       �	�9Nfc�A�#*

loss?&*<��.       �	5�Nfc�A�#*

loss�0P;2�m       �	�sOfc�A�#*

loss��D<0%       �	Pfc�A�#*

loss^=#C!       �	�Pfc�A�#*

loss���;��xh       �	4Qfc�A�#*

loss�yV;Ӕ5�       �	��Qfc�A�#*

loss=h9<�W4N       �	m�Rfc�A�#*

loss pK=s       �	%Sfc�A�#*

lossOT�<��        �	(�Sfc�A�#*

loss7Q.< p�M       �	d[Tfc�A�#*

loss�ב=��B�       �	�Tfc�A�#*

loss/��<oNXt       �	��Ufc�A�#*

lossQ�N=F3H       �	Z*Vfc�A�#*

lossz�39v��       �	��Vfc�A�#*

loss)i:�@Ȣ       �	�YWfc�A�#*

loss�EJ;��տ       �	��Wfc�A�#*

loss�"�;J@b$       �	�Xfc�A�#*

lossLx�<�L��       �	*Yfc�A�#*

loss��R<���h       �	�Yfc�A�#*

lossq4�<��$       �	vSZfc�A�#*

lossN
_<�[�       �	��Zfc�A�#*

loss��=��*o       �	��[fc�A�#*

loss3� <E�       �	,)\fc�A�#*

loss8��<��T9       �	e�\fc�A�#*

loss���;i�L       �	�c]fc�A�$*

lossD��;16m�       �	�^fc�A�$*

loss��<�       �	ɪ^fc�A�$*

loss=h�<�L��       �	�I_fc�A�$*

loss���<����       �	j�_fc�A�$*

lossѸ�<(��+       �	By`fc�A�$*

loss�w�<U��       �	�afc�A�$*

losskI<4|>k       �	 bfc�A�$*

lossJ|P;���       �	Z�bfc�A�$*

lossݳ�:tP)�       �	-?cfc�A�$*

lossl�;6�B       �	�cfc�A�$*

loss�T�<9DK{       �	�~dfc�A�$*

loss-�h=�`�b       �	Defc�A�$*

lossxn�;v4�,       �	�efc�A�$*

loss�=��=S       �	��ffc�A�$*

loss�}K;	v�<       �	j�gfc�A�$*

lossp�<���-       �	�phfc�A�$*

lossl�"<�n@�       �	�ifc�A�$*

loss��<J)�       �	��ifc�A�$*

lossL�i;�c(�       �	DMjfc�A�$*

loss��<�Q�       �	�jfc�A�$*

loss1ԃ<o���       �	-�kfc�A�$*

lossZ.`<�_�       �	�.lfc�A�$*

loss�F=�J$       �	\�lfc�A�$*

loss��;�7��       �	�amfc�A�$*

lossʘ;��4g       �	�mfc�A�$*

lossH�=P˧6       �	i�nfc�A�$*

loss�=|^�?       �	=)ofc�A�$*

loss��;Z�WM       �	p�ofc�A�$*

loss%V�<jyd       �	��pfc�A�$*

loss�/�;0N>�       �	�#qfc�A�$*

loss�\B<h�       �	1�qfc�A�$*

loss�s;	� �       �	�grfc�A�$*

loss�0x<A�B       �	�sfc�A�$*

loss4T0=9�Q}       �	�sfc�A�$*

loss�l�<���       �	q�tfc�A�$*

loss�l=W��       �	jufc�A�$*

loss,I�;v;�l       �	(*vfc�A�$*

loss�ʸ:�'�5       �	)�vfc�A�$*

lossE�=�1D�       �	�wfc�A�$*

lossE�X<��       �	s0xfc�A�$*

loss-�;ne��       �	��xfc�A�$*

loss/��<}b�"       �	�xyfc�A�$*

loss��+=ςϸ       �	g(zfc�A�$*

lossF� =Z���       �	��zfc�A�$*

loss���<T�3       �	7q{fc�A�$*

lossO5�:-��       �	�|fc�A�$*

loss�<P?��       �	C�|fc�A�$*

lossqy�<YB�H       �	h�~fc�A�$*

loss!�<B��       �	��fc�A�$*

loss��<Ðe�       �	�,�fc�A�$*

lossמ�<u-��       �	�ˀfc�A�$*

loss6�=	���       �	@j�fc�A�$*

loss���<���X       �	C �fc�A�$*

loss7M�:��9�       �	���fc�A�$*

lossV�p<�Y�       �	H�fc�A�$*

loss��=��?       �	�ރfc�A�$*

loss��V<�8       �	�s�fc�A�$*

loss��;ڧ��       �	��fc�A�$*

lossN��:Ij�%       �	ı�fc�A�$*

loss�<^Hl�       �	�U�fc�A�$*

loss��&;[@D7       �	,�fc�A�$*

loss_�.;�r��       �	덇fc�A�$*

loss�[�=E�P       �	"�fc�A�$*

loss�O@<JvZ;       �	˿�fc�A�$*

loss��:��c>       �	�Z�fc�A�$*

lossӺ�=펼
       �	��fc�A�$*

loss�;�T��       �	���fc�A�$*

loss���;v^��       �	�!�fc�A�$*

loss�0<� ƒ       �	�ڋfc�A�$*

lossJ��<�3r�       �	��fc�A�$*

loss)b�;欩�       �	.�fc�A�$*

loss���<�5Pw       �	���fc�A�$*

loss��r<�@�}       �	ɐ�fc�A�$*

lossHX�<����       �	�$�fc�A�$*

loss��<A�(�       �	�fc�A�$*

loss��R:�1�=       �	y\�fc�A�$*

lossֻL=��Ñ       �	��fc�A�$*

loss�L;���       �	Z��fc�A�$*

loss�9;V       �	w�fc�A�$*

loss^
=q��M       �	;��fc�A�$*

loss@ƒ:�m˭       �	xE�fc�A�$*

losss��<)�)�       �	הfc�A�$*

loss��=�8�       �	�z�fc�A�$*

lossh;�U�       �	��fc�A�$*

lossQ�1:�A�       �	7��fc�A�$*

loss���;Cr[�       �	�K�fc�A�$*

loss#7:U}��       �	��fc�A�$*

loss���<Ä       �	���fc�A�$*

loss�7;�)_       �	�fc�A�$*

losss�<��3�       �	���fc�A�$*

loss1%==��       �	�K�fc�A�$*

loss��g;�R��       �	�ߚfc�A�$*

loss׸�;���       �	~q�fc�A�$*

lossD҄:o���       �	��fc�A�$*

loss�=Ǖ�Y       �	���fc�A�$*

loss�:��ۤ       �	�;�fc�A�$*

loss���<����       �	k՝fc�A�$*

loss�<�D=�       �	"o�fc�A�$*

loss]&= D=       �	��fc�A�$*

loss9P�<w4T�       �	��fc�A�$*

loss�<8�?       �	3�fc�A�$*

loss#�;��E       �	�ܠfc�A�$*

loss\��<��St       �	�r�fc�A�$*

loss}��<�7SO       �	��fc�A�$*

loss��==��=:       �	;¢fc�A�$*

lossө�<�y�       �	#��fc�A�$*

lossHO�<���       �	��fc�A�$*

loss$~j<��)       �	ȴ�fc�A�$*

loss''�=��>       �	�J�fc�A�$*

loss;A�:�;�       �	��fc�A�$*

loss1��;�ʩ)       �	�y�fc�A�$*

loss\_=��\       �	�?�fc�A�$*

losso`�=�~��       �	/�fc�A�$*

lossύ"<L�8       �	�~�fc�A�$*

lossX:���*       �	!�fc�A�$*

loss���;���       �	8��fc�A�$*

loss�Z�<���       �	eQ�fc�A�$*

loss6A;�z�       �	��fc�A�$*

loss�+_;^�       �	��fc�A�$*

loss�K�<��
�       �	��fc�A�$*

loss�х=��n�       �	x��fc�A�$*

loss��&;�F�       �	iR�fc�A�$*

lossl =��!H       �	��fc�A�$*

loss\c�<���       �	�fc�A�$*

loss�*�;6���       �	A)�fc�A�$*

loss���9Z��j       �	ʯfc�A�$*

loss0�;?o        �	3l�fc�A�$*

loss	� =D�>       �	��fc�A�%*

lossowA<����       �	��fc�A�%*

loss���;\�|       �	�(�fc�A�%*

loss�n�7��L       �	1A�fc�A�%*

lossL�C:����       �	�n�fc�A�%*

lossS�<���       �	�:�fc�A�%*

loss�e�7�2��       �	�p�fc�A�%*

lossL�(:�=:       �	2Ǹfc�A�%*

loss��:�'�       �	�i�fc�A�%*

lossE�<��)       �	��fc�A�%*

lossK�;?�9I       �	���fc�A�%*

loss���:�ѯ       �	H3�fc�A�%*

loss���;�       �	�Żfc�A�%*

loss��<`G�       �	B\�fc�A�%*

loss��u;�S�       �	��fc�A�%*

loss:�H<s�WX       �	u��fc�A�%*

loss�~�<�O�       �	B�fc�A�%*

loss���<���       �	M�fc�A�%*

loss�<@H8�       �	�ÿfc�A�%*

lossC{<��       �	�`�fc�A�%*

loss��]=5��       �	��fc�A�%*

losst�g=���r       �	t��fc�A�%*

loss6	<?>re       �	eU�fc�A�%*

loss]oq=�q�       �	^��fc�A�%*

loss;�=)�%�       �	8.�fc�A�%*

loss��:�c�m       �	���fc�A�%*

lossq<YsU       �	�N�fc�A�%*

lossW0W<]xo       �	���fc�A�%*

loss穄<�2B       �	M��fc�A�%*

loss��=)��       �	�fc�A�%*

lossl�;�0�*       �	ɰ�fc�A�%*

loss���;�T�X       �	|F�fc�A�%*

loss�<VC       �	��fc�A�%*

loss� �<�<i       �	]��fc�A�%*

loss��<	�&m       �	O$�fc�A�%*

lossΧ;��%H       �	���fc�A�%*

lossv|<qv}�       �	��fc�A�%*

loss�ۀ:%��       �	B!�fc�A�%*

loss��.:T���       �	`?�fc�A�%*

loss�7R:�[ϖ       �	���fc�A�%*

loss��=y�MK       �	�{�fc�A�%*

loss4�=k��       �	)!�fc�A�%*

loss�!L=�)�       �	R��fc�A�%*

loss�A�<�:E       �	2s�fc�A�%*

loss���<L�&�       �	��fc�A�%*

loss|7P:M!o.       �	j��fc�A�%*

loss��;d�D�       �	�{�fc�A�%*

losso��:���m       �	1y�fc�A�%*

loss#)=�E��       �	4�fc�A�%*

loss���:[_�e       �	���fc�A�%*

losso��<�8��       �	���fc�A�%*

loss�@<��e       �	>A�fc�A�%*

loss&=2%��       �	���fc�A�%*

loss8��:K��       �	�m�fc�A�%*

loss?C:��       �	��fc�A�%*

loss�2*;
5Jd       �	���fc�A�%*

loss��;Wt�(       �	r6�fc�A�%*

loss��o<�7��       �	���fc�A�%*

loss�8�:k�A�       �	{g�fc�A�%*

loss��<v�-�       �	���fc�A�%*

lossM��<I!\�       �	���fc�A�%*

loss`�7;�""�       �	/5�fc�A�%*

loss���<R��       �	���fc�A�%*

loss�^�;��
       �	uu�fc�A�%*

loss��w<���       �	^�fc�A�%*

lossT�;3���       �	���fc�A�%*

loss�;���       �	u�fc�A�%*

lossכ&=��i       �	F�fc�A�%*

loss���<5�q�       �	��fc�A�%*

lossn@�<��,�       �	T:�fc�A�%*

loss�r";�
�\       �	c��fc�A�%*

loss�NG<�C��       �	,g�fc�A�%*

loss&��;ݭW       �	&��fc�A�%*

lossct>a�%�       �	h��fc�A�%*

loss$_K;��,'       �	�)�fc�A�%*

loss�W;���        �	+��fc�A�%*

lossa�=z���       �	�Q�fc�A�%*

lossMD�<8���       �	���fc�A�%*

loss\��<���|       �	5z fc�A�%*

loss#y�<��8       �	�fc�A�%*

loss;?c;"���       �	�fc�A�%*

loss�E:�z7       �	!@fc�A�%*

loss xh<�43t       �	��fc�A�%*

loss�D>='%}       �	z�fc�A�%*

loss ��<��G       �	t'fc�A�%*

lossZ�<�dj�       �	:�fc�A�%*

loss)�s=y��r       �	�nfc�A�%*

loss��H<�}��       �	hfc�A�%*

loss��I=nw>�       �	Нfc�A�%*

loss_�q<�GfJ       �	�>fc�A�%*

loss�"a<֙Z�       �	��fc�A�%*

lossJ*�<�M��       �	�fc�A�%*

loss��;ʓ�e       �	�!	fc�A�%*

lossC�=����       �		�	fc�A�%*

loss�� <m        �	�-fc�A�%*

lossa��;'��f       �	/�fc�A�%*

loss��I<��)	       �	@hfc�A�%*

loss6��;��C�       �	�	fc�A�%*

loss�h�<�6*3       �	�fc�A�%*

loss���;a"��       �	�Sfc�A�%*

loss�Y�;{��       �	qfc�A�%*

lossir<D��       �	��fc�A�%*

loss[s�<�1'$       �	�Cfc�A�%*

loss!A�=ӎD       �	y=fc�A�%*

loss$�=���       �	fc�A�%*

lossq��=ؠO�       �	��fc�A�%*

lossF�=���       �	!Yfc�A�%*

loss
-�:	�J       �	��fc�A�%*

loss��<]��       �	4�fc�A�%*

loss(�;I���       �	d�fc�A�%*

lossQ�<_��W       �	L3fc�A�%*

loss��S<�Z�       �	�fc�A�%*

loss�:$;�L��       �	�dfc�A�%*

loss;_';�"��       �	��fc�A�%*

loss)�?;M+�#       �	:�fc�A�%*

loss�@�;�VU       �	</fc�A�%*

loss��=��c       �	!�fc�A�%*

loss��;DۢL       �	mfc�A�%*

loss��=�ua"       �	�fc�A�%*

loss�Z;��       �	��fc�A�%*

loss���:r�q(       �	�Nfc�A�%*

lossM%�:v��       �	��fc�A�%*

loss�	�:h�       �	c~fc�A�%*

loss}i�<q��X       �	-%fc�A�%*

lossk�<��       �	"�fc�A�%*

loss�ɸ<zg>1       �	�` fc�A�%*

loss�S;�s�       �	�o!fc�A�%*

loss��;I��       �	?�"fc�A�%*

loss�;�<�iC       �	c#fc�A�&*

lossG�;'��       �	��#fc�A�&*

lossHI;�&       �	V�$fc�A�&*

loss�Q�=���|       �	�:%fc�A�&*

loss��<+���       �	$�%fc�A�&*

loss\_�;2��       �	~s&fc�A�&*

loss�M=k�f�       �	�'fc�A�&*

lossH�<3�0       �	v�'fc�A�&*

lossAi�<���       �	RE(fc�A�&*

loss �g=]�ۼ       �	�(fc�A�&*

loss&g <V�*n       �	�t)fc�A�&*

lossL�V<_7��       �	�*fc�A�&*

loss��6<�AV       �	z�*fc�A�&*

lossDu;��|�       �	S@+fc�A�&*

loss��;m�G.       �	s�+fc�A�&*

loss;<�M�)       �	�l,fc�A�&*

loss���<��[       �	-fc�A�&*

lossV^�<ul�       �	 �-fc�A�&*

loss��<쌜�       �	�+.fc�A�&*

loss]�=ZT%       �	�.fc�A�&*

loss�T�<B��(       �	/�/fc�A�&*

lossW">;�G;       �	NE0fc�A�&*

lossJ��;��^       �	�0fc�A�&*

loss<~<Nc�)       �	܀1fc�A�&*

loss�� <h\q       �	[%2fc�A�&*

lossC�<�>qm       �	��2fc�A�&*

loss���<��	       �	�c3fc�A�&*

loss
��;V�a       �	�4fc�A�&*

lossh;<�nA�       �	\�4fc�A�&*

loss��<����       �	wL5fc�A�&*

lossLy�<��˦       �	�O6fc�A�&*

loss3�~<�Q�[       �	��6fc�A�&*

lossMK<��'       �	��7fc�A�&*

loss�=sʧ�       �	�?8fc�A�&*

lossL��;m�@�       �	L�9fc�A�&*

loss��l<��U�       �	p(:fc�A�&*

loss��K<�       �	��:fc�A�&*

loss�<_��Y       �	g_;fc�A�&*

loss�F?9u��       �	r�;fc�A�&*

loss���;[       �	��<fc�A�&*

loss�G�<��        �	7=fc�A�&*

loss6��<��o~       �	��=fc�A�&*

loss?��<���t       �	�s>fc�A�&*

loss��%<�M��       �	,?fc�A�&*

loss�/<�MW       �	��?fc�A�&*

loss��<a�	�       �	�F@fc�A�&*

loss�;5G��       �	�@fc�A�&*

loss���<��5Q       �	�Afc�A�&*

loss֟�<�OG       �	S Bfc�A�&*

loss�Oo<j�}       �	��Bfc�A�&*

lossR�4;.��       �	]Cfc�A�&*

loss�ҥ;1���       �	��Cfc�A�&*

lossϝ�:�_t_       �	F�Dfc�A�&*

loss��<��,W       �	i:Efc�A�&*

lossx�:<�5�       �	��Efc�A�&*

loss�@�;2��?       �	�}Ffc�A�&*

loss�;��       �	� Gfc�A�&*

losswi~<Z^�F       �	E�Gfc�A�&*

loss2o;�B�$       �	K\Hfc�A�&*

lossX��<��       �	o�Hfc�A�&*

lossv݂=��l�       �	!�Ifc�A�&*

lossZ�n:�a�       �	�-Jfc�A�&*

loss�g9<����       �	�Jfc�A�&*

loss��,;[l$�       �	<kKfc�A�&*

loss�<�>�       �	�
Lfc�A�&*

loss$S<=�N{       �	ܷLfc�A�&*

loss�|#<�=�       �	��Mfc�A�&*

lossQ�K;љZ3       �	�"Nfc�A�&*

lossew�<��-�       �	ǼNfc�A�&*

lossjW�;��R�       �	�VOfc�A�&*

loss�;�3       �	�Pfc�A�&*

lossq�=���\       �	��Pfc�A�&*

loss��!<Z�cH       �	,Qfc�A�&*

loss�)<��l�       �	��Rfc�A�&*

loss*<����       �	6vSfc�A�&*

loss,�;0t�0       �	�Tfc�A�&*

loss��<[d       �	�Tfc�A�&*

loss��=<p.D�       �	tBUfc�A�&*

loss���<�;�G       �	��Ufc�A�&*

lossI�'<dˑ       �	lVfc�A�&*

loss��<�f�c       �	�Wfc�A�&*

loss_�=�$ �       �	O�Wfc�A�&*

loss9��<�~^       �	oHXfc�A�&*

loss��<��^       �	�Xfc�A�&*

loss�h�<cL�1       �	vYfc�A�&*

loss���;���       �	�Zfc�A�&*

lossE&�<��       �	@�Zfc�A�&*

lossi�J;���W       �	��[fc�A�&*

lossFP1<��W       �	�1\fc�A�&*

loss)"�=3<G       �	��\fc�A�&*

loss���<i�       �	�]]fc�A�&*

loss�@�=�y~       �	�^fc�A�&*

loss���;D�|       �	]�^fc�A�&*

loss�$:7i^       �	EJ_fc�A�&*

loss7R>a�       �	P�_fc�A�&*

loss8�O<���Q       �	hv`fc�A�&*

loss�=;<St�*       �	�afc�A�&*

loss���=���       �	��afc�A�&*

loss���:�       �	�Bbfc�A�&*

lossj·;��'	       �	�bfc�A�&*

loss6�}:bwY       �	�kcfc�A�&*

lossO��<X�)�       �	� dfc�A�&*

losslљ:��w       �	��dfc�A�&*

loss�(;��S       �	f.efc�A�&*

loss}��<ѡ�i       �	��efc�A�&*

loss�3;<�b       �	x}ffc�A�&*

loss�37:�帺       �	�gfc�A�&*

loss���<_��       �	�gfc�A�&*

loss�/�:n��       �	�hhfc�A�&*

loss�M<)A�       �	bifc�A�&*

loss��<H0�J       �	}�ifc�A�&*

loss	؝:e�	       �	�Tjfc�A�&*

loss_��=�Yw       �	r�jfc�A�&*

lossW�<)�       �	�kfc�A�&*

lossn�=)^��       �	'Mlfc�A�&*

loss��1=s�       �	`�lfc�A�&*

lossr�;�?�^       �	�mfc�A�&*

lossH� ;종        �	_Cnfc�A�&*

loss�8='Df�       �	��nfc�A�&*

lossv�<˔�       �	'�ofc�A�&*

lossZL;�s�       �	:pfc�A�&*

loss˞;�urQ       �	C�pfc�A�&*

loss$+D;&<�        �	{kqfc�A�&*

loss_��;v��       �	�rfc�A�&*

loss�ˆ=�i��       �	(�rfc�A�&*

loss�f	<Q�$�       �	�^sfc�A�&*

loss3�&<͸�7       �	�tfc�A�&*

loss�h.=ɲ�       �	z�tfc�A�&*

lossϢ<�ټ�       �	~vfc�A�'*

lossm��<��=�       �	f�vfc�A�'*

loss}d�<��:       �	Q�wfc�A�'*

loss��9c�X'       �	�yfc�A�'*

loss ]J=#�0       �	��yfc�A�'*

loss:��<�G�g       �	#{fc�A�'*

loss�܍<����       �	7|fc�A�'*

loss�J.=�U�<       �	o/}fc�A�'*

loss�۲;�"�       �	��}fc�A�'*

loss��<
�X�       �	k~fc�A�'*

loss��<�;�       �	�fc�A�'*

lossR��:�\0�       �	�fc�A�'*

lossʹ<�^�       �	�2�fc�A�'*

loss�;���       �	CȀfc�A�'*

lossh�;X+�       �	�}�fc�A�'*

loss�+�<7���       �	��fc�A�'*

loss
�0<��	       �	�fc�A�'*

loss*��<��       �	�^�fc�A�'*

loss��;MO��       �	���fc�A�'*

loss��H<<��       �	2��fc�A�'*

loss��5;[�"       �	�>�fc�A�'*

loss���:v#�       �	�݅fc�A�'*

lossɾ�;k�T�       �	�{�fc�A�'*

loss���;��Z       �	9�fc�A�'*

lossk=��z       �	㩇fc�A�'*

loss@˄=�)zZ       �	?�fc�A�'*

loss[��<��       �	�ڈfc�A�'*

loss� �=�βq       �	-z�fc�A�'*

loss��4=���       �	�fc�A�'*

loss�}�<4{�,       �	���fc�A�'*

loss��=(�i       �	�F�fc�A�'*

lossڌ�;�rw�       �	q�fc�A�'*

loss���<6�2       �	k��fc�A�'*

loss��6<�؀�       �	��fc�A�'*

loss�mx=-8��       �	�Ǎfc�A�'*

lossOS�<����       �	�a�fc�A�'*

loss���<��       �	��fc�A�'*

lossLJ�;�O��       �	��fc�A�'*

lossxٓ:�ݫ�       �	9+�fc�A�'*

loss\!�:<X8       �	�Ȑfc�A�'*

lossM�
<�А�       �	Dj�fc�A�'*

loss��<�9hc       �	T��fc�A�'*

lossכD;P	�:       �	���fc�A�'*

loss _;��B�       �	�(�fc�A�'*

loss֥<-r       �	�fc�A�'*

loss�^D<.D2       �	X�fc�A�'*

loss2=�п       �	6"�fc�A�'*

loss�_�<�yRJ       �	���fc�A�'*

lossĿ=2�F�       �	YM�fc�A�'*

loss�o
<+C-�       �	H�fc�A�'*

loss n�<^��I       �	d��fc�A�'*

loss��<+�
�       �	�B�fc�A�'*

lossMC�=���       �	��fc�A�'*

lossS�<�ښ       �	��fc�A�'*

loss�M';F�       �	!�fc�A�'*

loss�Y�;�G&       �	���fc�A�'*

loss���<�'N}       �	�X�fc�A�'*

loss�4p=/�0x       �	��fc�A�'*

loss��;����       �	T��fc�A�'*

loss%�<�t       �	{-�fc�A�'*

lossHۦ<��       �	�ɝfc�A�'*

lossF<XF�       �	*q�fc�A�'*

lossmR;�T��       �	��fc�A�'*

loss���:�G�M       �	S��fc�A�'*

lossV#<*��       �	Kv�fc�A�'*

loss(��;f�y       �	��fc�A�'*

loss�-<���       �	�ϡfc�A�'*

loss�׀<J��       �	�Ȣfc�A�'*

loss���<��8�       �	Ad�fc�A�'*

loss:X<�P�       �	� �fc�A�'*

loss��O<��Q%       �	ࢤfc�A�'*

loss��$<�J�       �	�?�fc�A�'*

lossx��;ɸ�}       �	��fc�A�'*

loss��=	��2       �	�}�fc�A�'*

loss<0i:�Gw�       �	�$�fc�A�'*

lossI�<�*�       �	M֧fc�A�'*

lossnr"<�y�0       �	逨fc�A�'*

loss�LH;��$�       �	Xp�fc�A�'*

lossL��;ԝ��       �	\U�fc�A�'*

lossM�:Q��       �	v��fc�A�'*

loss�c�<�i�       �	/��fc�A�'*

lossq��;�$b�       �	P�fc�A�'*

loss/��<LWmB       �	���fc�A�'*

lossu=�e�V       �	�fc�A�'*

loss��=�/�u       �	�Q�fc�A�'*

loss��A=X�L       �	M�fc�A�'*

lossWi ;ŷ�k       �	���fc�A�'*

lossΖ<R�       �	�J�fc�A�'*

loss���<�;       �	�fc�A�'*

lossJ��:�ĥ:       �	I��fc�A�'*

loss�;�N)J       �	}@�fc�A�'*

lossů�;�r��       �	��fc�A�'*

loss��s<�mJ       �	x�fc�A�'*

loss�C�< �_       �	�fc�A�'*

loss'�<��@       �	'��fc�A�'*

lossȎ�<i>?       �	�˵fc�A�'*

loss}�%<�8B�       �	��fc�A�'*

loss��;�i       �	5�fc�A�'*

loss�Al;H�       �	{�fc�A�'*

loss�j4<��       �	1��fc�A�'*

loss�a;�-j       �	]N�fc�A�'*

loss�e�:���       �	Z�fc�A�'*

loss�+':�NE       �	֋�fc�A�'*

lossT�;?6e       �	P9�fc�A�'*

loss`o=��o       �	wֻfc�A�'*

lossS
:R�Tk       �	�r�fc�A�'*

lossLG�=�LB       �	{�fc�A�'*

loss���<aNT�       �	y��fc�A�'*

loss{Oj< U&�       �	J�fc�A�'*

lossAE�;��u       �	��fc�A�'*

loss&0�<�D��       �	ˢ�fc�A�'*

loss�~Z; ȗ       �	GU�fc�A�'*

loss��=D<�m       �	{��fc�A�'*

loss�k�;>�T       �	��fc�A�'*

loss�S=���       �	�@�fc�A�'*

loss6�(;ax�-       �	'O�fc�A�'*

loss�I�;�A�       �	x��fc�A�'*

lossCjR;��G       �	Ҍ�fc�A�'*

lossTp<dð�       �	�'�fc�A�'*

loss��3;��/       �	��fc�A�'*

lossV�<5Z�%       �	�a�fc�A�'*

lossX�<��       �	���fc�A�'*

loss+%;%�9       �	��fc�A�'*

loss�w�;�9       �	+3�fc�A�'*

loss��=���       �	���fc�A�'*

losst3�;B/��       �	�n�fc�A�'*

loss龜<A��z       �	���fc�A�'*

loss��o<�3�	       �	u>�fc�A�'*

lossa*< tR�       �	Y��fc�A�(*

loss&g<�
�t       �	���fc�A�(*

lossX��<|J�       �	�fc�A�(*

loss���<����       �	_��fc�A�(*

lossjߝ9�ū�       �	�H�fc�A�(*

loss�!=���       �	���fc�A�(*

loss��@<�       �	�fc�A�(*

loss���<���       �	�-�fc�A�(*

loss���;B}��       �	:��fc�A�(*

loss�J;��|       �	3j�fc�A�(*

loss��<?�6�       �	g�fc�A�(*

loss���<g?�       �	t��fc�A�(*

lossX�<y@�       �	.V�fc�A�(*

loss@��:o���       �	#��fc�A�(*

loss���<e���       �	��fc�A�(*

loss���<���       �	�?�fc�A�(*

losst>�:�a       �	���fc�A�(*

lossS�<����       �	Bv�fc�A�(*

loss�˘;u9�X       �	��fc�A�(*

loss8<�<��:       �	ۿ�fc�A�(*

loss��:ɞ��       �	�a�fc�A�(*

loss���<�|       �	?��fc�A�(*

lossҌR<h��       �	��fc�A�(*

loss]�>N�B       �	y<�fc�A�(*

loss��D; ��       �	���fc�A�(*

lossQ�@=j��       �	���fc�A�(*

loss�Q�<�I�       �	EG�fc�A�(*

lossC,�;6�\        �	���fc�A�(*

loss�T�<zkx       �	�u�fc�A�(*

loss�(<�J       �	��fc�A�(*

loss�6�=h�_       �	��fc�A�(*

loss�o�<�Q��       �	�=�fc�A�(*

loss�":��3       �	 �fc�A�(*

loss}�	= ���       �	���fc�A�(*

loss���<j       �	�g�fc�A�(*

losst�g<pߠ�       �	��fc�A�(*

loss`�;C�
       �	���fc�A�(*

loss��;"G�D       �	�h�fc�A�(*

loss[20<�Hm�       �	�
�fc�A�(*

loss&}:�Y�?       �	Զ�fc�A�(*

loss<d<յ��       �	�_�fc�A�(*

loss|D*<��[�       �	:�fc�A�(*

loss]�<wgp       �	]��fc�A�(*

loss�#Y<B�-       �	�r�fc�A�(*

loss
_�<1��       �	��fc�A�(*

lossy�<"�E       �	i��fc�A�(*

loss�.<oqH       �	�E�fc�A�(*

loss�=����       �	��fc�A�(*

loss��;�n�       �	p}�fc�A�(*

loss��=�?��       �	]�fc�A�(*

lossw��<'��       �	u��fc�A�(*

loss7<�_I�       �	J�fc�A�(*

loss�J:0��S       �	 ��fc�A�(*

loss}�< ���       �	J|�fc�A�(*

lossF�M<���       �	%�fc�A�(*

loss��<�Yy�       �	E��fc�A�(*

lossr^=AG       �	�R�fc�A�(*

lossM=�3�       �	S��fc�A�(*

loss���;,��       �	���fc�A�(*

loss$N=���v       �	�'�fc�A�(*

lossI��;��x�       �	,��fc�A�(*

lossN�J=D��R       �	�m�fc�A�(*

loss4`;��?*       �	��fc�A�(*

loss��=I�"       �	���fc�A�(*

loss�x$=�Xo�       �	"4�fc�A�(*

loss�!<��݂       �	L��fc�A�(*

loss/��<��1       �	���fc�A�(*

loss��&<8���       �	|��fc�A�(*

loss�}3<Bu�N       �	QL�fc�A�(*

loss�c�:�Z       �	��fc�A�(*

loss1/�;}��       �	��fc�A�(*

loss/d<���       �	�*�fc�A�(*

loss�w�;~g=       �	}��fc�A�(*

lossk�;s��        �	���fc�A�(*

lossi�<*NV       �	�y�fc�A�(*

loss�a�<%�;       �	N`�fc�A�(*

loss���<��/       �	���fc�A�(*

loss$� :�`�       �	S��fc�A�(*

loss�`o<07$P       �	�- fc�A�(*

loss��	<�M�       �	t� fc�A�(*

loss�h�<��Lz       �	{gfc�A�(*

loss�@8<�k]U       �	�	fc�A�(*

lossIT�:���       �	�fc�A�(*

loss���<����       �	1Efc�A�(*

loss�0<-s(�       �	/�fc�A�(*

loss*�<���       �	�rfc�A�(*

loss4l<=�-.\       �	fc�A�(*

loss4!�;?W�{       �	��fc�A�(*

lossA@>�`�:       �	�Dfc�A�(*

loss��;*lV4       �	d�fc�A�(*

loss��9=_S˒       �	$|fc�A�(*

lossY=�js�       �	�fc�A�(*

loss�7�<�]M       �	K�fc�A�(*

loss�%;�*�o       �	D	fc�A�(*

losse�q<k���       �	��	fc�A�(*

loss�P<z�       �	�q
fc�A�(*

loss���<|���       �	�fc�A�(*

lossJ�:AIl8       �	4�fc�A�(*

loss���;�ajZ       �	{Ifc�A�(*

loss��<�+�       �	��fc�A�(*

loss-�;��|�       �	�fc�A�(*

loss���<�
H�       �	�jfc�A�(*

loss+b=�C�       �	�fc�A�(*

loss6<��J=       �	R�fc�A�(*

lossС>�P-       �	.Sfc�A�(*

loss�
=k�{       �	1�fc�A�(*

lossF�d=w�RT       �	Q�fc�A�(*

lossU�<�ԑc       �	`#fc�A�(*

loss|=�<�$�       �	zfc�A�(*

loss�Ps;���       �	�fc�A�(*

lossw�7<�߽Z       �	%�fc�A�(*

loss�/;����       �	�Hfc�A�(*

loss��<���-       �	@�fc�A�(*

loss�tl;R��}       �	�|fc�A�(*

loss��Z<�1�       �	ofc�A�(*

lossF&�<��l�       �	��fc�A�(*

lossZ�C<�I%       �	[Cfc�A�(*

lossC��<^%�y       �	��fc�A�(*

loss���<=@�       �	�fc�A�(*

loss�{>;�M��       �	�1fc�A�(*

lossr0�;�3�       �	�fc�A�(*

lossx�	=�o��       �	ifc�A�(*

lossx�<�؊�       �	Wfc�A�(*

lossd='�܋       �	/�fc�A�(*

loss�!=� G�       �	�=fc�A�(*

loss��<���-       �	��fc�A�(*

loss�><���       �	�ufc�A�(*

lossӾ�<�C\       �	�!fc�A�(*

loss�@�<~��        �	�fc�A�)*

lossR:�<ل��       �	KV fc�A�)*

loss��<�0       �	�� fc�A�)*

loss
T=r>5Z       �	�!fc�A�)*

loss�K�;V�_@       �	�#"fc�A�)*

lossb�<4��^       �	g�"fc�A�)*

loss�F9;��6       �	�R#fc�A�)*

loss���<5�Qu       �	S�#fc�A�)*

lossL� =�K#       �	�$fc�A�)*

loss��r;NF       �	�%fc�A�)*

loss�B�<�       �	K�%fc�A�)*

loss2d�<��1       �	�H&fc�A�)*

loss!K=c.؛       �	��&fc�A�)*

loss���;&y�'       �	��'fc�A�)*

loss���:#�2�       �	(fc�A�)*

loss�<�a�       �	�(fc�A�)*

loss�FK=�&`\       �	�T)fc�A�)*

lossh�;�/�f       �	��)fc�A�)*

loss��!<q7l�       �		�*fc�A�)*

lossEo<U��       �	}!+fc�A�)*

lossA��:� �Z       �	g�+fc�A�)*

losso�3</��       �	XY,fc�A�)*

loss��c;��       �	��,fc�A�)*

loss���=����       �	[�-fc�A�)*

loss���;���       �	5.fc�A�)*

lossm��;8 �O       �	��.fc�A�)*

loss&�>=^�;       �	c/fc�A�)*

loss��<�F       �	�/fc�A�)*

loss��;�2�       �	l�0fc�A�)*

loss�:��	r       �	X;1fc�A�)*

loss/��;��z�       �	D�1fc�A�)*

losst_<��       �	�w2fc�A�)*

loss�.�<A���       �	r3fc�A�)*

loss�Cn=t��U       �	ȱ3fc�A�)*

loss�+v;��       �	�K4fc�A�)*

lossO&)<����       �	��4fc�A�)*

loss7��=G��       �	��5fc�A�)*

lossɰ�<��5       �	'6fc�A�)*

lossr��;2o�,       �	��6fc�A�)*

lossqĦ;jڄ�       �	Eb7fc�A�)*

loss�&�< �7R       �	8fc�A�)*

loss�=,;���       �	<�8fc�A�)*

loss��<�X��       �	:<9fc�A�)*

loss�R�<���{       �	\�9fc�A�)*

loss�,�<?K�       �	�{:fc�A�)*

loss�S�:c���       �	�;fc�A�)*

loss���<��       �	��;fc�A�)*

loss��C;�:�       �	�J<fc�A�)*

loss/==jo       �	��<fc�A�)*

lossTZ�;�<��       �	�v=fc�A�)*

loss�F�<b�0       �	^>fc�A�)*

loss�<�x��       �	3�>fc�A�)*

loss��<6�r�       �	FD?fc�A�)*

loss�mP=Ž�$       �	��?fc�A�)*

loss�4�:�\�L       �	2�@fc�A�)*

lossEG`:��\_       �	{2Afc�A�)*

loss\7%:B]�(       �	|�Afc�A�)*

loss<�;�S�       �	�oBfc�A�)*

loss�f
=����       �	�Cfc�A�)*

loss�ez<�X�       �	�Cfc�A�)*

loss)�K:�-       �	�TDfc�A�)*

loss�j�;	�Ԗ       �	��Dfc�A�)*

loss��;���       �	ސEfc�A�)*

lossd�c:;Z�/       �	)Ffc�A�)*

loss=c;-�jG       �	��Ffc�A�)*

lossATx;���!       �	�oGfc�A�)*

loss�W^<��       �	FHfc�A�)*

loss�qq<�@{       �	¥Hfc�A�)*

lossۇ�;�pZ       �	[?Ifc�A�)*

lossf�<c2�t       �	��Ifc�A�)*

loss��N=�>4       �	�qJfc�A�)*

loss6;.}�p       �	�	Kfc�A�)*

loss�E?;m��        �	�Kfc�A�)*

loss�N�:��,�       �	|CLfc�A�)*

lossl$t<�J�^       �	��Lfc�A�)*

loss.��:r��       �	�vMfc�A�)*

loss���;V���       �	Nfc�A�)*

lossz�<�zi*       �	?�Nfc�A�)*

lossS2;(yv�       �	%ZOfc�A�)*

lossׄ�;O��       �	�Pfc�A�)*

loss��d<�_��       �	��Pfc�A�)*

loss<�=��f       �	zqQfc�A�)*

lossYނ;H���       �	�Rfc�A�)*

loss�:U��n       �	|�Rfc�A�)*

loss:�; �       �	o�Sfc�A�)*

lossQ�:�ӳ       �	�>Tfc�A�)*

loss�G9��Q�       �	i�Tfc�A�)*

loss?�[;*�v�       �	|{Ufc�A�)*

loss;lZ:���       �	Vfc�A�)*

loss]�:ѹ*       �	�BWfc�A�)*

loss	|�;��f       �	�*Xfc�A�)*

loss��:�v       �	��Xfc�A�)*

loss�߆:�E�\       �	HjYfc�A�)*

loss �4=!��       �	EZfc�A�)*

loss���8���       �	V�Zfc�A�)*

lossb+�9��       �	{\fc�A�)*

loss�]�:s&��       �	��\fc�A�)*

lossZQ;Y/�       �	F]fc�A�)*

loss�,;�nE]       �	��]fc�A�)*

loss�j�:���       �	�{^fc�A�)*

loss���9<ˌi       �	3_fc�A�)*

loss�>n�N       �	[�_fc�A�)*

loss�f�9Ɯ�       �	2X`fc�A�)*

loss =Q�       �	��`fc�A�)*

lossM�<�w/�       �	��afc�A�)*

lossD%);-�9!       �	�5bfc�A�)*

loss�.�;pib�       �	-�bfc�A�)*

lossT�;QW)       �	Qjcfc�A�)*

loss�xq;�2       �	-	dfc�A�)*

loss��<�;4�       �	Q�dfc�A�)*

loss4J;�       �	>Aefc�A�)*

loss�D�;��       �	o�efc�A�)*

loss1e<<=Ӽy       �	�ffc�A�)*

loss��M<�S:       �	 gfc�A�)*

loss-n!<���e       �	�gfc�A�)*

loss,T�;{���       �	]hfc�A�)*

loss)jk<��X�       �	o�hfc�A�)*

loss�<{K؈       �	�ifc�A�)*

loss��;����       �	A-jfc�A�)*

loss��:K���       �	?�jfc�A�)*

loss�d�<�~?       �	�kfc�A�)*

loss�ϸ;ti|       �	�@lfc�A�)*

lossW�:o@G       �	��lfc�A�)*

loss)\);z���       �	`xmfc�A�)*

loss�2�;���~       �	"nfc�A�)*

loss)1S=,�^�       �	Q�nfc�A�)*

loss���9X��       �	PQofc�A�)*

loss!�|:�77       �	��ofc�A�)*

loss��q=���-       �	�pfc�A�**

loss�u-<Zi�       �	aqfc�A�**

lossx'=pwK       �	ɯqfc�A�**

loss�BB=��       �	�Orfc�A�**

lossx
�:u�C       �	'�tfc�A�**

loss��
=q�D@       �	{ufc�A�**

lossz��=\��       �	�vfc�A�**

lossC :��]       �	�vfc�A�**

loss�;y'W~       �	%\wfc�A�**

lossQ;-�K       �	]xfc�A�**

loss#�;z�rA       �	��xfc�A�**

loss:k
=�s��       �	{zfc�A�**

loss4�=;���       �	G{fc�A�**

loss���;r?gM       �	��{fc�A�**

loss�2#<B>       �	4�|fc�A�**

loss82<_��q       �	&�}fc�A�**

loss4�y<�       �	��~fc�A�**

loss��J<6�       �	��fc�A�**

loss�d;<#       �	q�fc�A�**

loss�=I|�       �	�Ёfc�A�**

loss�u�<��2       �	��fc�A�**

loss�KS;>���       �	=��fc�A�**

loss͸<��       �	SZ�fc�A�**

loss��:}U�       �	
�fc�A�**

lossL�<z�	�       �	��fc�A�**

loss��<�̇K       �	z��fc�A�**

loss�j<��}       �	�A�fc�A�**

loss8#=Qn        �	�ڰfc�A�**

loss&cW;uԟ�       �	>u�fc�A�**

loss*�a<�P       �	M�fc�A�**

losshY<�       �	���fc�A�**

loss�g.<YH��       �	�O�fc�A�**

loss8F�<v��+       �	��fc�A�**

lossS��<E��       �	���fc�A�**

loss��;�i��       �	��fc�A�**

loss���:��v�       �	�|�fc�A�**

loss;w<�Q�       �	^�fc�A�**

lossMh*=��       �	.��fc�A�**

loss8��<:�r       �	E�fc�A�**

loss��<{�2�       �	�]�fc�A�**

loss���;#��       �	�:�fc�A�**

loss�4�95`�E       �	U�fc�A�**

loss]�9<��
       �	�D�fc�A�**

lossʛ<,3H       �	�d�fc�A�**

lossOw�;J2�       �	s�fc�A�**

lossC��<NyN�       �	;R�fc�A�**

lossD��<f��       �	���fc�A�**

loss�1J<�#Y�       �	S��fc�A�**

loss�<=E�       �	=��fc�A�**

lossp�<���       �	�E�fc�A�**

loss��<���C       �	���fc�A�**

loss�T=Fs�       �	�z�fc�A�**

loss�(#<W�6�       �	���fc�A�**

lossw�<�YV       �	N�fc�A�**

loss�?�:m��R       �	9��fc�A�**

lossל�;��-�       �	ٙ�fc�A�**

loss�ۡ;���       �	3�fc�A�**

loss���;�c4       �	O��fc�A�**

loss��h<$�8       �	t�fc�A�**

loss��<�a�       �	�fc�A�**

loss��,=���       �	��fc�A�**

loss��;���P       �	�M�fc�A�**

loss	�<��       �	8�fc�A�**

loss=��<��f        �	J��fc�A�**

loss1��<��+       �	�T�fc�A�**

lossO�=���       �	 ��fc�A�**

loss�A(=�n�p       �	Z��fc�A�**

loss� �:��|�       �	L�fc�A�**

loss^'<Mn��       �	T��fc�A�**

loss�A�<��d�       �	D�fc�A�**

loss+<sP�       �	���fc�A�**

loss�5�<�"/�       �	.r�fc�A�**

loss8=D�ɀ       �	��fc�A�**

loss��/;V�{       �	���fc�A�**

loss�#�9ٮU       �	�^�fc�A�**

loss�á;��t       �	���fc�A�**

loss�~i=��]�       �	���fc�A�**

loss���;�a%�       �	���fc�A�**

loss�=a}       �	*8�fc�A�**

losse	�<��       �	*��fc�A�**

loss��;'43       �	U��fc�A�**

loss���:�:��       �	-�fc�A�**

loss���:���       �	���fc�A�**

loss�C�;��]�       �	\q�fc�A�**

loss�
=g�ڰ       �	��fc�A�**

loss��U=�	:�       �	��fc�A�**

loss���;���       �	t@�fc�A�**

loss��9�d�l       �	���fc�A�**

loss�M=���       �	�q�fc�A�**

loss��/<��43       �	m�fc�A�**

loss��;�8�#       �	���fc�A�**

lossys;4��r       �	�M�fc�A�**

lossna<���	       �	���fc�A�**

loss���<(w��       �	���fc�A�**

loss<֋<��m       �	H7�fc�A�**

lossy�=�LӶ       �	��fc�A�**

lossJF�<��G       �	�g�fc�A�**

loss��<���       �	e�fc�A�**

loss�<n:��r       �	���fc�A�**

loss �d==�P�       �	N+�fc�A�**

lossbH�<�k0       �	A��fc�A�**

loss�l<<R��_       �	"p�fc�A�**

loss�7<<��h�       �	8k�fc�A�**

loss�<�RLv       �	�fc�A�**

loss�=,oQ�       �	B��fc�A�**

loss�2�;5
a       �	8K�fc�A�**

loss�,c<X8��       �	���fc�A�**

loss%@�;Y�1T       �	i��fc�A�**

loss#�#:��       �	w.�fc�A�**

loss�|d<���*       �	��fc�A�**

loss�f�<�Rr�       �	<f�fc�A�**

lossQ�<恏       �	c��fc�A�**

loss+Ó=�Z�       �	���fc�A�**

lossM<���E       �	q:�fc�A�**

loss��U<ŗ�v       �	���fc�A�**

loss}o=\�+       �	N~�fc�A�**

lossr3�<+$a       �	��fc�A�**

loss�U<wŗ�       �	��fc�A�**

loss]�<���       �	��fc�A�**

loss�/�<�b�H       �	#�fc�A�**

lossq6:;��       �	���fc�A�**

loss�e�=~��       �	�h�fc�A�**

losswb=)�       �	>�fc�A�**

loss�82<~Ƃ�       �	���fc�A�**

loss|�|;0,G       �	:<�fc�A�**

loss��;P�
�       �	���fc�A�**

loss���;�}J       �	m�fc�A�**

loss���<g(P\       �	�O�fc�A�**

loss�&;/k�       �	_��fc�A�+*

losse�<=^q�       �	���fc�A�+*

loss�$�;�u�       �	�#�fc�A�+*

lossIN�<y��O       �	f��fc�A�+*

loss馪<�[`       �	&S�fc�A�+*

losszeC=,?�Y       �	�L�fc�A�+*

loss���;���z       �	��fc�A�+*

loss�x=�y6+       �	��fc�A�+*

loss�/�<N���       �	"T�fc�A�+*

loss���<(z�       �	���fc�A�+*

lossm��;ʒ��       �	���fc�A�+*

loss׌E<�Y#       �	�H�fc�A�+*

loss�G};+�       �	��fc�A�+*

loss^k�;a�2�       �	"��fc�A�+*

loss�5=ˆ�       �	�$�fc�A�+*

loss[��<V��       �	��fc�A�+*

loss��;0]�       �	�� fc�A�+*

loss�`�;��~       �	�+fc�A�+*

lossjg=�FR       �	U�fc�A�+*

lossOyf;�~4^       �	�Zfc�A�+*

loss��'=�t�m       �	��fc�A�+*

loss���:�:Ӆ       �	�fc�A�+*

loss��$=���       �	EIfc�A�+*

loss���<�pyt       �	��fc�A�+*

loss[bk;�T{U       �	�fc�A�+*

loss�N�<#�^�       �	_)fc�A�+*

lossL�<+��       �	��fc�A�+*

loss��2<} 
       �	nfc�A�+*

loss�g=���s       �	Vfc�A�+*

loss\q<;���|       �	:�fc�A�+*

loss��Y;��}�       �	�L	fc�A�+*

loss4l�<��       �	9�	fc�A�+*

loss=�=���       �	9�
fc�A�+*

loss���<Q��       �	�?fc�A�+*

loss�Ϣ;���t       �	��fc�A�+*

loss��<qY��       �	�fc�A�+*

loss-i�:,�       �	�&fc�A�+*

loss:��:���       �	x�fc�A�+*

loss�Ȋ<epii       �	fjfc�A�+*

loss�Oh;�4I�       �	�fc�A�+*

loss/(<p~�       �	n�fc�A�+*

lossi��=L���       �	Dfc�A�+*

loss2��<6� /       �	b�fc�A�+*

loss�
�;X��       �	�fc�A�+*

loss�};oW��       �	�#fc�A�+*

loss4�;
���       �	��fc�A�+*

loss��K;a�       �	�^fc�A�+*

loss3"M;�Ϥ       �	��fc�A�+*

lossA�[;�
�       �	��fc�A�+*

loss�j0<;�f�       �	NAfc�A�+*

loss�<���       �	�fc�A�+*

loss"��;�M��       �	͏fc�A�+*

lossC<i4�       �	�7fc�A�+*

loss�<���       �	��fc�A�+*

lossM��<s�>�       �	�fc�A�+*

loss��<��_�       �	fc�A�+*

loss�y&;�9;       �	��fc�A�+*

loss��	=0q�       �	dfc�A�+*

loss��;�o       �	_fc�A�+*

lossth�;�W?	       �	�fc�A�+*

loss�\:uL��       �	,Ifc�A�+*

loss��;;��t       �	=�fc�A�+*

lossz;�(�       �	ގfc�A�+*

loss���:��       �	�)fc�A�+*

lossь�<4��V       �	��fc�A�+*

loss6t�;�1�d       �	A�fc�A�+*

loss��`;�%       �	}! fc�A�+*

loss��;<&	G�       �	(� fc�A�+*

lossq2;�:��       �	U!fc�A�+*

loss*.<�Ց�       �	��!fc�A�+*

loss�2b;x{�@       �	=�"fc�A�+*

loss�إ;�ՐU       �	�#fc�A�+*

loss�~=�Z-       �		�#fc�A�+*

loss6��<	T��       �	R_$fc�A�+*

lossd�Q<��       �	��$fc�A�+*

lossj�<���       �	��%fc�A�+*

loss���:{-�       �	s-&fc�A�+*

loss��u9g>�       �	9�&fc�A�+*

loss���;$?K1       �	�l'fc�A�+*

loss��P:J��`       �	�(fc�A�+*

loss�}3;d1i       �	R�(fc�A�+*

loss�5;�]�B       �	�4)fc�A�+*

loss��;�kh       �	��)fc�A�+*

lossF�_<�&�       �	J`*fc�A�+*

loss��=�	       �	f�*fc�A�+*

loss��<���       �	)�+fc�A�+*

loss��8<ː@s       �	31,fc�A�+*

loss%�'<G���       �	��,fc�A�+*

loss��=�i       �	�k-fc�A�+*

lossZ��;�
�       �	�.fc�A�+*

lossi�R<���       �	�.fc�A�+*

loss��:g�]�       �	�4/fc�A�+*

loss���;u��U       �	B�/fc�A�+*

loss��3;`Ό       �	�e0fc�A�+*

lossn<�<���       �	��0fc�A�+*

loss�E=b �Q       �	��1fc�A�+*

lossߊ�;��-       �	�,2fc�A�+*

loss�X�=�X��       �	r�2fc�A�+*

lossV�;wAeX       �	�Z3fc�A�+*

loss�<:;�"�        �	]�3fc�A�+*

loss�!�;!��       �	��4fc�A�+*

lossN/;���       �	�45fc�A�+*

loss�M�:x��       �	��5fc�A�+*

loss��<b��       �	ro6fc�A�+*

lossE`<[��       �	�7fc�A�+*

loss��4=,�wS       �	��7fc�A�+*

loss�4<��r       �	�28fc�A�+*

loss�.r;u$��       �	�8fc�A�+*

loss�OM<��z�       �	D�9fc�A�+*

loss���<P��"       �	�;fc�A�+*

loss���;��1       �	V(<fc�A�+*

loss �4<���       �	W�=fc�A�+*

loss
�G=�'�       �	h�>fc�A�+*

loss�>=�{�9       �	 :?fc�A�+*

loss��"=�4S       �	�?fc�A�+*

lossI��=�2��       �	|@fc�A�+*

loss��P=�m��       �	�Afc�A�+*

loss:O�;��s       �	��Afc�A�+*

loss_k<Z �)       �	|_Bfc�A�+*

loss/C7;��}`       �	��Bfc�A�+*

loss�yF<�G-       �	��Cfc�A�+*

loss|]2:Bp�$       �	�RDfc�A�+*

losszB�<�.-       �	��Dfc�A�+*

loss�|�;×n       �	�Efc�A�+*

lossWƳ;�_��       �	�9Ffc�A�+*

loss?�};��       �	��Ffc�A�+*

loss�Ψ;I=F       �	jiGfc�A�+*

loss��(<�"]       �	��Gfc�A�+*

loss���;�؜�       �	�Hfc�A�,*

loss�oN<G��       �	�9Ifc�A�,*

loss�1�;�В        �	�Ifc�A�,*

loss8�G<�߉;       �	��Jfc�A�,*

loss�ԑ;��       �	J$Kfc�A�,*

lossͅh<���       �	-�Kfc�A�,*

loss!/�<��       �	�wLfc�A�,*

loss��=|n�       �	Mfc�A�,*

lossZ��;h       �	q�Mfc�A�,*

lossF'<GV�-       �	�@Nfc�A�,*

loss�:;�;b�       �	6Ofc�A�,*

loss�.;��#       �	�Ofc�A�,*

loss-�f=����       �	�\Pfc�A�,*

loss�U<o<K       �	��Pfc�A�,*

loss��; )@�       �	ˠQfc�A�,*

loss�X;<�e       �	�6Rfc�A�,*

loss�q5;�p��       �	��Rfc�A�,*

lossx�=����       �	�sSfc�A�,*

loss���;����       �	-Tfc�A�,*

lossj�9<'�w�       �	��Tfc�A�,*

loss�(;��*�       �	=Ufc�A�,*

loss=�,<�g��       �	5�Ufc�A�,*

loss:�:/��)       �	�iVfc�A�,*

losso��::8u       �	?Wfc�A�,*

loss3e�;�8/       �	^�Wfc�A�,*

loss.2:o�lO       �	�CXfc�A�,*

loss�F�<2��(       �	]�Xfc�A�,*

loss�g�=��W�       �	�uYfc�A�,*

loss���<�-��       �	PZfc�A�,*

loss�5);	=�=       �	��Zfc�A�,*

loss��;8F�n       �	YM[fc�A�,*

loss^=��?&       �	��[fc�A�,*

loss�W�;k�K       �	\fc�A�,*

loss!��=
I�       �	1$]fc�A�,*

lossZU�:�2�       �	��]fc�A�,*

loss��b=0N��       �	[^fc�A�,*

loss_��<ڂ9�       �	�^fc�A�,*

losst<�H�n       �	��_fc�A�,*

loss��<�R �       �	�@`fc�A�,*

loss��:��       �	U�`fc�A�,*

lossH9Q;���       �	�qafc�A�,*

loss�AY;;��       �	�bfc�A�,*

lossӯ�<��s       �	ڨbfc�A�,*

loss���<�Pf�       �	�Fcfc�A�,*

lossv=-=���0       �	��cfc�A�,*

loss���<5��       �	�xdfc�A�,*

loss̄�;7W��       �	efc�A�,*

loss{{�<�Q�U       �	�efc�A�,*

loss��c=4e�       �	SBffc�A�,*

loss$ �9�4e&       �	��ffc�A�,*

loss��=
��       �	xgfc�A�,*

loss��;�\˸       �	�hfc�A�,*

lossi�*<�0��       �	D�hfc�A�,*

loss��<�y1       �	tBifc�A�,*

loss@<=�       �	��ifc�A�,*

loss��+<�4�p       �	�sjfc�A�,*

loss���:*r��       �	tkfc�A�,*

loss�CP=��C\       �	��kfc�A�,*

loss:P5<��x	       �	JElfc�A�,*

lossD�:���       �	a�lfc�A�,*

lossR�;�FqD       �	�mfc�A�,*

loss:Vh:c@ŕ       �	�nfc�A�,*

lossn��:+OL�       �	 �nfc�A�,*

lossD�;�=�       �	%=ofc�A�,*

lossʞ�=��N=       �	j�ofc�A�,*

loss��9 ��       �	qupfc�A�,*

loss;=	�N_       �	Zqfc�A�,*

loss�;�<�H�9       �	��qfc�A�,*

loss�[<��       �	Irfc�A�,*

loss��h;��7       �	��rfc�A�,*

loss!�<��B       �	�sfc�A�,*

loss7�;@u��       �	.tfc�A�,*

loss��</�a�       �	x�tfc�A�,*

loss��=�_�       �	�Pufc�A�,*

lossͶ=�*��       �	�ufc�A�,*

loss�@f;��s       �	�vfc�A�,*

loss��;�i�       �	zwfc�A�,*

loss�K�<�SJL       �	>�wfc�A�,*

loss	s<�Q��       �	|{xfc�A�,*

loss_֨;�[�L       �	�yfc�A�,*

lossxV<Xc�       �	�yfc�A�,*

loss�a<����       �	�Mzfc�A�,*

loss�j1<����       �	��zfc�A�,*

loss	�<s*�D       �	y{fc�A�,*

loss֪=[�W7       �	�|fc�A�,*

lossۖ�;�)�/       �	q�|fc�A�,*

lossLLt;�)��       �	RD}fc�A�,*

lossj�;��}h       �	�}fc�A�,*

loss�U<�V�S       �	T�~fc�A�,*

loss��;�/z�       �	Gfc�A�,*

loss�5=��خ       �	0�fc�A�,*

loss�=[׀�       �	^��fc�A�,*

lossS=���       �	�`�fc�A�,*

loss�W�=�7�       �	 ��fc�A�,*

loss\1�;���{       �	t��fc�A�,*

lossZ�<�:�V       �	�8�fc�A�,*

loss3R<�(�       �	�σfc�A�,*

loss��</i@J       �	�j�fc�A�,*

loss��J<}���       �	e�fc�A�,*

loss6}<G�~       �	���fc�A�,*

loss�I�<� �       �	'M�fc�A�,*

loss�^:gg��       �	��fc�A�,*

loss��=��e�       �	J�fc�A�,*

loss,m=�3�       �	��fc�A�,*

lossM_�9�/��       �	!��fc�A�,*

loss�<�ai�       �	�D�fc�A�,*

loss|39=7W       �	�މfc�A�,*

lossߏ�<�%	�       �	�s�fc�A�,*

loss34�:��       �	W�fc�A�,*

loss�Z�:�h?�       �	���fc�A�,*

loss�ؽ:���P       �	�.�fc�A�,*

loss�Y;�+X       �	\ƌfc�A�,*

lossWhq<TU�       �	K[�fc�A�,*

loss��>n!��       �	���fc�A�,*

loss��P<ו+�       �	狎fc�A�,*

loss6';<63�P       �	m�fc�A�,*

lossH�<z�e�       �	
܏fc�A�,*

loss)_(<�e�I       �	5|�fc�A�,*

lossCey<}�d_       �	 �fc�A�,*

loss1�Q;�       �	D��fc�A�,*

loss���;Y$#       �	���fc�A�,*

loss��?<���z       �	rO�fc�A�,*

loss ͖<����       �	��fc�A�,*

lossmK�:��y       �		��fc�A�,*

loss��F<1�F       �	�0�fc�A�,*

loss�� ;ڽ|       �	�˕fc�A�,*

loss�0�<�u�       �	d@�fc�A�,*

loss���:u^%�       �	��fc�A�,*

loss��<a��'       �	2��fc�A�-*

loss�6�;��E       �	_&�fc�A�-*

loss�l�;G�2       �	3��fc�A�-*

loss�/k<*�       �	�b�fc�A�-*

loss���<��&       �	� �fc�A�-*

loss�[%=��o�       �	'��fc�A�-*

lossI	4=>��n       �	�A�fc�A�-*

loss��"=�Lʥ       �	��fc�A�-*

lossU�<x       �	4��fc�A�-*

loss�<>�>       �	)�fc�A�-*

loss��<����       �	�ɟfc�A�-*

loss���<2��       �	Xr�fc�A�-*

lossR��<e$�       �	��fc�A�-*

loss憽=-�|�       �	3¡fc�A�-*

loss���<Y�K       �	{f�fc�A�-*

loss5��:*�<       �	��fc�A�-*

loss��	<��X\       �	9��fc�A�-*

loss�}"=
t8       �	�1�fc�A�-*

loss�ݒ<Πe�       �	�ʤfc�A�-*

loss��d<��6�       �	Me�fc�A�-*

loss�7�<����       �	K�fc�A�-*

loss31<B](       �	V��fc�A�-*

loss��X=	wG       �	�:�fc�A�-*

lossS�"=3��       �	G�fc�A�-*

loss���;�y�       �	ۨfc�A�-*

loss���;�ڬz       �	�o�fc�A�-*

loss�|�<�>y"       �	q�fc�A�-*

loss�ME<F�6       �	��fc�A�-*

loss�BY;�       �	�/�fc�A�-*

loss�׵9愼       �	xҫfc�A�-*

loss1v/;#���       �	%y�fc�A�-*

loss��2=j���       �	`�fc�A�-*

lossQ�;@ӊ%       �	"­fc�A�-*

loss�5=��~�       �	:]�fc�A�-*

lossa��<���       �	8�fc�A�-*

losspS�<�4'�       �	D��fc�A�-*

loss�.S=#1��       �	.�fc�A�-*

loss�M59(���       �	���fc�A�-*

loss}o:� \�       �	�P�fc�A�-*

loss|<|��       �	��fc�A�-*

loss4'�<ş�[       �	�fc�A�-*

loss��H<U|��       �	��fc�A�-*

loss�z;{�
       �	��fc�A�-*

lossM�)<��M�       �	�6�fc�A�-*

loss�0T:=p8       �	�ߵfc�A�-*

loss�R�<۶m}       �	��fc�A�-*

lossy�<1bD�       �	�.�fc�A�-*

loss���<UcC       �	1�fc�A�-*

loss�^�<�\_�       �	k*�fc�A�-*

lossA��<+R�       �	�һfc�A�-*

loss��;<���(       �	P��fc�A�-*

loss]�:="���       �	�*�fc�A�-*

loss(��<���       �	 �fc�A�-*

loss�Q<��{[       �	뫾fc�A�-*

loss�2�<�Uf*       �	�{�fc�A�-*

loss��;�c�       �	6\�fc�A�-*

loss�)�<Ċ9       �	��fc�A�-*

loss��9�<�       �	���fc�A�-*

loss(R�:��m�       �	f/�fc�A�-*

lossd�<e��       �	���fc�A�-*

loss��V<��       �	8i�fc�A�-*

loss{��:��d�       �	_�fc�A�-*

loss�Ӕ<QE"       �	&��fc�A�-*

loss��^9�|�u       �	^I�fc�A�-*

lossWL�=�SR       �	��fc�A�-*

loss?�<�s�j       �	���fc�A�-*

lossO=<�	3       �	9(�fc�A�-*

lossN	U<q�E�       �	���fc�A�-*

loss<�;��       �	�t�fc�A�-*

loss�YI;���       �	w�fc�A�-*

loss&�;-�ͻ       �	2��fc�A�-*

loss=��;��Ё       �	�H�fc�A�-*

loss�;�;���       �	���fc�A�-*

lossٍ;%���       �	o�fc�A�-*

lossT��;u��       �	S �fc�A�-*

loss|��=d!��       �	���fc�A�-*

loss�'=�3F       �	{O�fc�A�-*

loss�W�<���       �	*��fc�A�-*

loss�O=�^�C       �	ګ�fc�A�-*

lossRp;�l>%       �	SA�fc�A�-*

loss�b�:R�N#       �	0��fc�A�-*

loss)I="Rhw       �	�m�fc�A�-*

loss�j&=���H       �	�fc�A�-*

loss�_&<��       �	���fc�A�-*

lossƏ�=���       �	jK�fc�A�-*

lossX;5y�       �	3��fc�A�-*

lossa,;�-6       �	qu�fc�A�-*

loss
�a=�L��       �	�
�fc�A�-*

loss��=!��%       �		��fc�A�-*

loss'�<��       �	~;�fc�A�-*

loss�ng=�$ܐ       �	���fc�A�-*

loss?��<e[��       �	<l�fc�A�-*

loss���:WU�       �	 ��fc�A�-*

loss_o�<�GÕ       �	؜�fc�A�-*

loss�M�:l�}       �	�3�fc�A�-*

loss�<d�vG       �	y��fc�A�-*

loss0<![?       �	p�fc�A�-*

loss��\;1��       �	B'�fc�A�-*

lossX4�<�J�i       �	x��fc�A�-*

loss��<���}       �	˃�fc�A�-*

loss��3<��;       �	A�fc�A�-*

loss�.�;Q_       �	���fc�A�-*

loss�=;���&       �	M��fc�A�-*

lossl��:�
(�       �	o.�fc�A�-*

lossS�v=� y�       �	���fc�A�-*

loss���;��y       �	Ii�fc�A�-*

loss� =�e0       �	�
�fc�A�-*

loss=B;�h�       �	X��fc�A�-*

lossvP.;}�T�       �	G�fc�A�-*

lossC$x=0dT�       �	���fc�A�-*

loss��H;rp
�       �	w��fc�A�-*

lossE
�=�a�q       �	K �fc�A�-*

loss���:{�u	       �	��fc�A�-*

lossq��:�3��       �	�O�fc�A�-*

loss��<Ai�       �	 ��fc�A�-*

loss/6;s_�F       �	���fc�A�-*

lossA�)=ڏ�
       �	v�fc�A�-*

loss��?;�h�       �	ɮ�fc�A�-*

loss͊=e��>       �	-Z�fc�A�-*

loss�Y�;��N        �	J��fc�A�-*

loss�;i<��S�       �	���fc�A�-*

loss�E�<��7,       �	}Y�fc�A�-*

lossDT�<��x�       �	��fc�A�-*

lossϴ�</�2�       �	9��fc�A�-*

loss�X�:��       �	+M�fc�A�-*

loss���<�,�i       �	���fc�A�-*

lossr�`;��g       �	_�fc�A�-*

loss��_;8p�       �	��fc�A�-*

loss��<c���       �	��fc�A�.*

loss,qI<��8       �	�H�fc�A�.*

lossZ�'<���       �	;��fc�A�.*

loss���<��-�       �	�v�fc�A�.*

lossj��<��>W       �	�fc�A�.*

loss�ѝ;����       �	ߨ�fc�A�.*

loss��#<�r��       �	�A�fc�A�.*

loss���:+�D�       �	��fc�A�.*

loss���;� �       �	6x�fc�A�.*

loss,'
<����       �	s�fc�A�.*

loss�7�;nX�       �	���fc�A�.*

loss.b=4 �       �	�B�fc�A�.*

loss+�;���       �	f��fc�A�.*

loss[U�<���R       �	U��fc�A�.*

loss��:�]�;       �	�~�fc�A�.*

loss��5;�☪       �	��fc�A�.*

loss���;���\       �	7��fc�A�.*

loss��:$YI#       �	@�fc�A�.*

loss��<dm��       �	���fc�A�.*

loss��D<v�M       �	���fc�A�.*

loss=�;h�Z�       �	���fc�A�.*

loss���<\�<       �	M��fc�A�.*

loss$$?<�H(i       �	��fc�A�.*

loss��;M��       �	�1�fc�A�.*

lossO�B<�?       �	u��fc�A�.*

loss!��8٦��       �	g�fc�A�.*

loss�(q;&+G�       �	��fc�A�.*

loss:�n<y#U�       �	���fc�A�.*

loss�s�;�{�K       �	�4 fc�A�.*

loss��<�5��       �	E� fc�A�.*

loss\h=%ؗD       �	_zfc�A�.*

loss�e<�y       �	�=fc�A�.*

loss�<�kL�       �	g�fc�A�.*

loss��d:�մ�       �	�lfc�A�.*

loss���;��Z%       �	�fc�A�.*

loss]Q�:۟�O       �	�fc�A�.*

loss\U�<t�͙       �	�@fc�A�.*

lossSU�<O��+       �	+�fc�A�.*

loss��:��        �	�ofc�A�.*

loss �<���+       �	_fc�A�.*

loss�jj=��       �	��fc�A�.*

loss_�={Ft�       �	�;fc�A�.*

loss7s�;� �       �	��fc�A�.*

loss�M�:᳆�       �	�}	fc�A�.*

loss�ni<���       �	'
fc�A�.*

loss�: �ʼ       �	K�
fc�A�.*

lossF�P9Ќ�       �	_Efc�A�.*

lossT�;͋,l       �	f�fc�A�.*

loss� �:�N�       �	�nfc�A�.*

loss�K�:n+w�       �	dfc�A�.*

lossF�;�d�#       �	(�fc�A�.*

loss��Q9�"�S       �	3fc�A�.*

loss[�:J�u�       �	C�fc�A�.*

lossj�p<�>��       �	�`fc�A�.*

loss6S�6x��       �	��fc�A�.*

loss�T;'�z]       �	��fc�A�.*

loss3��9ʝ��       �	|+fc�A�.*

lossDZ;�L��       �	��fc�A�.*

loss))<����       �	�fc�A�.*

lossm��:4I�       �	Dfc�A�.*

lossQ��<J��#       �	G�fc�A�.*

lossm��<��       �	(Cfc�A�.*

loss�֘9����       �	�fc�A�.*

lossV�<�M4�       �	�fc�A�.*

loss�j<,�        �	�Hfc�A�.*

loss���<�i�       �	��fc�A�.*

loss{�0<Dv�       �	~fc�A�.*

loss�
�<�	�       �	3�fc�A�.*

loss�.�<Xs�       �	�{fc�A�.*

lossr{L<L�m�       �	�fc�A�.*

loss��&=-z�       �	K�fc�A�.*

loss� ; 3�       �	�Mfc�A�.*

loss���;R�#       �	��fc�A�.*

lossր�=J�F       �		�fc�A�.*

loss���<�X�       �	�)fc�A�.*

lossX� ;�Ӕ�       �	��fc�A�.*

lossCe�; ;*       �	hfc�A�.*

losss�;ّ��       �	�fc�A�.*

loss��=�q       �	��fc�A�.*

loss۬#<�w'r       �	-@ fc�A�.*

loss�m,=܋��       �	�� fc�A�.*

loss�J[<���v       �	%v!fc�A�.*

lossH��:fF�Y       �	�"fc�A�.*

losso��;�a}�       �	]�"fc�A�.*

loss#�<�~ء       �	2:#fc�A�.*

lossOݥ;u+�       �	��#fc�A�.*

lossa�9�>g�       �	Bw$fc�A�.*

loss-��<��8R       �	�%fc�A�.*

loss���<Kl��       �	��%fc�A�.*

loss��r;U��:       �	�K&fc�A�.*

loss�p�<;���       �	��&fc�A�.*

lossVU�;4�'2       �	2�'fc�A�.*

loss4�<�)rl       �	�((fc�A�.*

loss��\;���v       �	�(fc�A�.*

loss];�Y       �	5])fc�A�.*

loss�
;t1��       �	K*fc�A�.*

loss�=!<$a�C       �	�*fc�A�.*

loss��5;`�ѝ       �	�7+fc�A�.*

loss��<�d�4       �	��+fc�A�.*

loss��'=g�R�       �	,�,fc�A�.*

loss)�7<��R       �	�N-fc�A�.*

loss�<�=n��       �	=�-fc�A�.*

loss�I�:���O       �	��.fc�A�.*

lossCJ�:G�*�       �	�P/fc�A�.*

loss�à;��p�       �	C�/fc�A�.*

lossPL�=\�˗       �	�y0fc�A�.*

loss��:�:       �	01fc�A�.*

loss�=�_m       �	f�1fc�A�.*

lossX�=���q       �	�72fc�A�.*

loss� �;K���       �	��2fc�A�.*

loss��[<�NH       �	�f3fc�A�.*

lossW��;QM a       �	z 4fc�A�.*

loss��<��S       �	��4fc�A�.*

lossp�=�]{�       �	�Tfc�A�.*

lossi��;6&'K       �	?Ufc�A�.*

loss��t<:��z       �	A�Ufc�A�.*

lossJh�;��       �	�sVfc�A�.*

loss�(�;��       �	�Wfc�A�.*

loss� <��r�       �	3�Wfc�A�.*

lossU<" [�       �	1DXfc�A�.*

loss![f;�YT       �	��Xfc�A�.*

loss�8N<��$�       �	�wYfc�A�.*

loss���<�GZ�       �	�Zfc�A�.*

loss��;�*;       �	��Zfc�A�.*

loss��f=⻭       �	�U[fc�A�.*

lossRD<�-.       �	E�[fc�A�.*

loss��]<����       �	{�\fc�A�.*

loss��;�ʖ       �	n]fc�A�.*

loss�N�<F�       �	ɮ]fc�A�/*

lossE-�9��vE       �	�K^fc�A�/*

lossw�:��8�       �	P�^fc�A�/*

loss�S<#a�g       �	yx_fc�A�/*

loss��6=NBmI       �	�`fc�A�/*

loss7��<�.��       �	��`fc�A�/*

loss�=$��       �	�@afc�A�/*

lossbg9��8&       �	6�afc�A�/*

lossa�=�A�       �	��bfc�A�/*

lossD�=���+       �	�cfc�A�/*

loss��;�f-       �	��cfc�A�/*

loss�[=:�]       �	MJdfc�A�/*

loss[D�;O�f�       �	��dfc�A�/*

loss]c0<�.�       �	J}efc�A�/*

loss�ڱ;�ş       �	�ffc�A�/*

lossr�;�Q�       �	P�ffc�A�/*

loss�;;����       �	Cgfc�A�/*

loss�0<<?;       �	+�gfc�A�/*

lossk,�<��a�       �	Svhfc�A�/*

lossҖi;�3.       �	�ifc�A�/*

loss���<=E�       �	?�ifc�A�/*

loss��:>kv.       �	�Pjfc�A�/*

loss"�=�[a%       �	��jfc�A�/*

loss���<�JW       �	�kfc�A�/*

loss�tA=>+ϙ       �	�lfc�A�/*

loss�k<Rc�S       �	��lfc�A�/*

lossc�;[H��       �	RGmfc�A�/*

loss�5=�E       �	&�mfc�A�/*

loss��<p�m       �	itnfc�A�/*

loss�L�;�7�       �	Eofc�A�/*

lossT9<|�Q�       �	��ofc�A�/*

loss�}`<���F       �	Fpfc�A�/*

loss���;��L       �	�pfc�A�/*

loss:n�;n
�       �	�tqfc�A�/*

loss�;9̆K       �	�	rfc�A�/*

lossD��<yb	       �	�rfc�A�/*

loss#΂<��~       �	w0sfc�A�/*

loss iz<�m       �	e�sfc�A�/*

loss���= ��$       �	`Ytfc�A�/*

loss��;w�       �	5�tfc�A�/*

lossq��9+)       �	.�ufc�A�/*

losss=;�j��       �	k(vfc�A�/*

loss�p�9@��       �	f�vfc�A�/*

loss�;���       �	-`wfc�A�/*

loss�̇<����       �	��wfc�A�/*

loss.3�<��t�       �	�\yfc�A�/*

lossc��;��v       �	rzfc�A�/*

loss�=E:S�J�       �	�R{fc�A�/*

loss���<5n�       �	.�|fc�A�/*

loss
�H;�N�       �	�\}fc�A�/*

loss���;i;��       �	�`~fc�A�/*

loss� L;�'�j       �	U�~fc�A�/*

loss1��;��v.       �	��fc�A�/*

lossZ8o<��+       �	y��fc�A�/*

lossLNh<Lh��       �	2��fc�A�/*

loss�)�:��-       �	��fc�A�/*

loss&:6=F��m       �	?ăfc�A�/*

loss�[�=�]O       �	||�fc�A�/*

lossɐ�;R|ځ       �	�H�fc�A�/*

loss��1<�T�       �	�{�fc�A�/*

loss�ټ;&��       �	���fc�A�/*

loss;K':o�#�       �	��fc�A�/*

loss�, <'�s       �	���fc�A�/*

loss�9<����       �	1�fc�A�/*

loss>�;ɋ�G       �	�1�fc�A�/*

loss��"=0�g       �	��fc�A�/*

loss�u	=�)7}       �	���fc�A�/*

loss�!P<;�       �	���fc�A�/*

loss��<��+$       �	BB�fc�A�/*

lossZ�:�5ȝ       �	|`�fc�A�/*

loss3�e=u���       �	Q�fc�A�/*

loss�S�;u3&�       �	��fc�A�/*

loss�v�<�T�w       �	���fc�A�/*

loss 8<|O>�       �	h�fc�A�/*

loss@��<�S{       �	�I�fc�A�/*

lossVԒ<��_
       �	kd�fc�A�/*

lossN'�<��g�       �	E�fc�A�/*

lossO";.ӕ~       �	���fc�A�/*

loss�;l�v�       �	�%�fc�A�/*

lossjn�;n��       �	˂�fc�A�/*

loss�G:��P       �	f��fc�A�/*

lossH4<��       �	uX�fc�A�/*

loss��=0c�       �	�4�fc�A�/*

lossoV�=�(f-       �	Ҝfc�A�/*

loss��)<�m��       �	d�fc�A�/*

lossfzQ;�|#       �	T��fc�A�/*

lossؐ�;�m
�       �	�fc�A�/*

lossQ�<���       �	:>�fc�A�/*

lossr�u<&��       �	�fc�A�/*

loss��&=�t;       �	�[�fc�A�/*

loss��<��\�       �	���fc�A�/*

loss�W<��       �	���fc�A�/*

loss,�y<Č�F       �	�8�fc�A�/*

loss�}<�y�f       �	3ܥfc�A�/*

loss�:uEA�       �	Jz�fc�A�/*

loss	�<�$�b       �	&�fc�A�/*

loss]�<�F       �	.ȧfc�A�/*

loss#�;���       �	�f�fc�A�/*

lossT�X<B2kF       �	��fc�A�/*

loss�s�;3��w       �	ϡ�fc�A�/*

lossڻ�8D��i       �	�֪fc�A�/*

loss��`=~Vh       �	�w�fc�A�/*

lossSl<X:y�       �	��fc�A�/*

losst <�b��       �	ͮ�fc�A�/*

lossZ<<��G�       �	K�fc�A�/*

loss�;�e8       �	��fc�A�/*

loss��<W[g       �	O��fc�A�/*

loss;�:=0r��       �	-�fc�A�/*

loss[W=�˒       �	�¯fc�A�/*

lossď:��=S       �	�˰fc�A�/*

lossE�;bp��       �	>\�fc�A�/*

loss1;���       �	��fc�A�/*

loss*�=5L�       �	)�fc�A�/*

loss�VW<6�N       �	L��fc�A�/*

loss <{<A�       �	t%�fc�A�/*

loss���9L�y1       �	ȴfc�A�/*

loss'\�<쟥S       �	�d�fc�A�/*

loss X�;�>�H       �	��fc�A�/*

loss��;I�:�       �	��fc�A�/*

loss�0�<{�       �	z6�fc�A�/*

lossh�;3��       �	BϷfc�A�/*

loss��5<���       �	���fc�A�/*

loss�C(<��)       �	���fc�A�/*

loss�~;ƶ�       �	*�fc�A�/*

loss���;p�Hx       �	z�fc�A�/*

lossV�(;(�       �	.�fc�A�/*

loss��;���       �	Nռfc�A�/*

loss��;��$       �	؝�fc�A�/*

loss� =�a6�       �	9G�fc�A�0*

loss�s�<�^��       �	��fc�A�0*

lossHA�<�@��       �	��fc�A�0*

lossRS�;���o       �	U�fc�A�0*

loss��)<3+��       �	���fc�A�0*

lossO2 <�fJ>       �	BB�fc�A�0*

loss$�<?���       �	.��fc�A�0*

loss�:/�(�       �	G��fc�A�0*

loss���;�b�d       �	XS�fc�A�0*

loss3��<���       �	���fc�A�0*

loss%(A<�%@�       �	�fc�A�0*

loss�n;me~�       �	��fc�A�0*

lossN��;��       �	R�fc�A�0*

loss�!';�(�y       �	1��fc�A�0*

loss�x�<�v_}       �	���fc�A�0*

lossv�<ѯ^       �	���fc�A�0*

loss[�1;k}�N       �	�:�fc�A�0*

loss�8�<��"�       �	���fc�A�0*

loss��<��j       �	�r�fc�A�0*

lossM��;⺱�       �	n�fc�A�0*

loss}Y<�D}       �	Ĳ�fc�A�0*

losss�<+R4       �	���fc�A�0*

loss��<t(z       �	�N�fc�A�0*

lossX��9ŇU�       �	���fc�A�0*

lossz�v=Ϲ��       �	���fc�A�0*

loss��<J�n�       �	8�fc�A�0*

loss-�:;��       �	R��fc�A�0*

loss��<Pсc       �	���fc�A�0*

loss��:<�xr�       �	��fc�A�0*

loss
�<��       �	9��fc�A�0*

loss�;?�&       �	�^�fc�A�0*

loss��-=����       �	���fc�A�0*

loss�;�)�       �	 ��fc�A�0*

lossx�)=�9܌       �	�V�fc�A�0*

loss܅�=���       �	���fc�A�0*

loss,1�=��+       �	��fc�A�0*

loss�}<���       �	�K�fc�A�0*

loss�|=90���       �	���fc�A�0*

lossI�B<�?�B       �	��fc�A�0*

loss��|;�\�       �	�2�fc�A�0*

lossH�g;��)|       �	R��fc�A�0*

loss��<Ea�}       �	t}�fc�A�0*

lossXt:�=%�       �	��fc�A�0*

loss]A�;9�        �	5��fc�A�0*

lossD�=��-�       �	MK�fc�A�0*

loss1�;��       �	L��fc�A�0*

loss� �;�F��       �	���fc�A�0*

loss4� <���       �	�0�fc�A�0*

loss �<��       �	��fc�A�0*

loss��:W�9A       �	�h�fc�A�0*

loss.)�;�x       �	��fc�A�0*

loss �1=!lW,       �	���fc�A�0*

lossZT<[>�%       �	�a�fc�A�0*

loss�js<�       �	���fc�A�0*

loss�I=F:�       �	]��fc�A�0*

loss��<�؏A       �	�#�fc�A�0*

loss�K�;��g^       �	,��fc�A�0*

loss� <=��5       �	�Q�fc�A�0*

lossJ�.<~�!       �	���fc�A�0*

loss��=�$��       �	ڌ�fc�A�0*

loss��};�w��       �	9%�fc�A�0*

loss���;��       �	:��fc�A�0*

loss�-g;w���       �	m�fc�A�0*

loss�I�<g�o@       �	��fc�A�0*

loss��='I��       �	���fc�A�0*

lossi�<�C܀       �	�2�fc�A�0*

loss���;�++e       �	��fc�A�0*

loss% ;f        �	<��fc�A�0*

lossO�r=1��c       �	7m�fc�A�0*

loss��'<b��       �	��fc�A�0*

loss�e�<:��Q       �	F��fc�A�0*

loss=D=x��`       �	~Q�fc�A�0*

loss�݊=��       �	���fc�A�0*

loss�_J=�?�_       �	<��fc�A�0*

lossɴ<[�U       �	! �fc�A�0*

loss�,�<�;�       �	��fc�A�0*

lossj��<�
�       �	c`�fc�A�0*

lossmd;���`       �	��fc�A�0*

loss*��;ѶC       �	��fc�A�0*

lossz��;�M��       �	��fc�A�0*

loss�t�<�Q@       �	���fc�A�0*

loss��;[-�       �	��fc�A�0*

loss�*�;탪�       �	���fc�A�0*

loss6��<-k�       �	�^�fc�A�0*

lossnh�<Ge��       �	���fc�A�0*

loss�<;�OZ       �	��fc�A�0*

losszT�:c��       �	@�fc�A�0*

loss]�l;�ϴw       �	���fc�A�0*

loss�s=�g��       �	�}�fc�A�0*

lossߍ;��,       �	��fc�A�0*

loss�t*<�߭<       �	���fc�A�0*

loss��>< D       �	`��fc�A�0*

lossԂ<���       �	�}�fc�A�0*

losst�=A��v       �	��fc�A�0*

loss��<rmt�       �	���fc�A�0*

loss� �<>��       �	ge�fc�A�0*

loss��@;),6�       �	% fc�A�0*

loss��"<��ͧ       �	� fc�A�0*

loss�r�;K��V       �	]6fc�A�0*

loss�Cu;N[�       �	9�fc�A�0*

losso�[<�h�y       �	�jfc�A�0*

loss%0]<5�z#       �	fc�A�0*

loss�'<֚��       �	�fc�A�0*

lossM;�b�       �	�Ffc�A�0*

loss]�;G�`D       �	qYfc�A�0*

loss�,�<T�\�       �	Y�fc�A�0*

loss��@=���       �	ݚfc�A�0*

loss�*R=���i       �	�;fc�A�0*

loss�1Q:e��$       �	�fc�A�0*

loss�'�;6;S       �	�ufc�A�0*

lossd��:�P'       �	6#	fc�A�0*

loss�n�:�P�       �	z�	fc�A�0*

loss���;C�#�       �	�d
fc�A�0*

loss��e;���x       �	�fc�A�0*

loss�y;J�~k       �	�jfc�A�0*

loss�1�=��i�       �	fc�A�0*

loss���<pi��       �	8�fc�A�0*

loss��;�<��       �	�Kfc�A�0*

loss:��<d|       �	��fc�A�0*

loss�T<l�\       �	��fc�A�0*

loss�	Z<!cE       �	I.fc�A�0*

loss�/�<��DH       �	��fc�A�0*

lossIF(;U�       �	�efc�A�0*

lossu�=M���       �	Ffc�A�0*

loss��<���       �	��fc�A�0*

lossm{�<�M�       �	�Efc�A�0*

loss��:<2KK       �	��fc�A�0*

loss�a�<;O�       �	��fc�A�0*

loss��=�2l       �	%fc�A�1*

loss�;J<�ҭ       �	��fc�A�1*

loss�<P��y       �	iqfc�A�1*

loss�;<^�ͽ       �	Ifc�A�1*

loss�p�<$&��       �	E�fc�A�1*

loss;Ǹ<έ       �	�dfc�A�1*

loss���;Ԩ��       �	t
fc�A�1*

lossܛ�;���       �	p�fc�A�1*

loss�ӌ<)M�l       �	7�fc�A�1*

loss8�:\��       �	^Jfc�A�1*

loss�1�:����       �	�\fc�A�1*

losszN�;a�i�       �	� fc�A�1*

loss7B�;$q��       �	� fc�A�1*

loss$��<g�       �	�E!fc�A�1*

loss��;�        �	�!fc�A�1*

loss�2�<�$�       �	^�"fc�A�1*

loss~�;ԕ�       �	�!#fc�A�1*

lossr[�<L��       �	��#fc�A�1*

loss��<�@=5       �	B_$fc�A�1*

loss�;��ތ       �	��$fc�A�1*

loss��:�)j�       �	ݗ%fc�A�1*

lossԅ�;��\�       �	~7&fc�A�1*

loss�}:�<��       �	g�&fc�A�1*

loss�<�;��O       �	\r'fc�A�1*

loss̣=5��       �	�(fc�A�1*

loss�J<z��z       �	��(fc�A�1*

lossDDK>m_S&       �	0G)fc�A�1*

loss�=�]z�       �	`�)fc�A�1*

loss�9;<%C�       �	�*fc�A�1*

lossՑ<9Xx       �	d#+fc�A�1*

loss��;��W�       �	f�+fc�A�1*

loss]X@;����       �	�],fc�A�1*

loss)��<���9       �	@�,fc�A�1*

loss��(='       �	Y�-fc�A�1*

loss�y=y1y       �	�s.fc�A�1*

loss��T=�f�       �	
/fc�A�1*

loss-͞;	>��       �	j�/fc�A�1*

loss#t�;�bxQ       �	�1fc�A�1*

loss�:j�2�       �	`�1fc�A�1*

loss$8�;�
��       �	&T2fc�A�1*

lossO�a<\s��       �	#�2fc�A�1*

loss��<��       �	U�3fc�A�1*

loss�3;�r�m       �	F4fc�A�1*

lossl�;��       �	Z�4fc�A�1*

loss�i<<�d�g       �	l�5fc�A�1*

loss��<��Ex       �	�36fc�A�1*

loss|H.<f�N#       �	��6fc�A�1*

loss�6�<yq0c       �	Ts7fc�A�1*

lossܑ�<A�T�       �	�8fc�A�1*

loss��p<�}�y       �	з8fc�A�1*

lossq�4=�T�       �	�c9fc�A�1*

loss��;�N�       �	&�9fc�A�1*

loss�.�:�jV�       �	�:fc�A�1*

loss�S�<ݭ�1       �	Y�;fc�A�1*

lossr�;�vֵ       �	�\<fc�A�1*

lossզ;�U�y       �	�9=fc�A�1*

loss���<׸�5       �	{�=fc�A�1*

loss�8<�ź�       �	A�>fc�A�1*

loss%{8=�c�l       �	P�?fc�A�1*

lossO�5<�Ր       �	�p@fc�A�1*

loss?e�;䟦�       �	�Afc�A�1*

lossCXj9��lt       �	�EBfc�A�1*

lossc��<���       �	zCfc�A�1*

loss2EW=����       �	�Dfc�A�1*

loss��;;���       �	s�Dfc�A�1*

loss�Z</�M�       �	��Efc�A�1*

loss�v;ܥ;       �	u�Ffc�A�1*

lossP<���       �	�/Gfc�A�1*

loss�|�:O�$       �	��Gfc�A�1*

loss��=��|�       �	�Hfc�A�1*

loss�Ĥ:��       �	O<Ifc�A�1*

loss�*;�Bt�       �	V�Ifc�A�1*

loss[��:�w��       �	ȲJfc�A�1*

loss�+�<P��       �	YKfc�A�1*

loss;��<9Rm@       �	Lfc�A�1*

loss�z;3v8�       �	��Lfc�A�1*

loss-��<���       �	(`Mfc�A�1*

loss]�H;!�ee       �	�Nfc�A�1*

lossW��<�"p�       �	�Nfc�A�1*

lossEpF;�x       �	�Ofc�A�1*

loss�<�DH       �	�]Pfc�A�1*

lossHVW<���[       �	a�Pfc�A�1*

loss�?�<�<|       �	��Qfc�A�1*

lossz+|<$���       �	�<Rfc�A�1*

loss�7Q<f:sk       �	|eSfc�A�1*

loss�QR;A��G       �	"RTfc�A�1*

loss��<r�W       �	��Tfc�A�1*

lossoG�:�&x       �	3�Ufc�A�1*

loss��=���       �	�{Vfc�A�1*

loss�/=��U       �	�Wfc�A�1*

loss&z�;ȹS�       �	��Wfc�A�1*

loss�x�;��       �	�`Xfc�A�1*

loss���<���       �	��Xfc�A�1*

lossY�;���       �	{�Yfc�A�1*

loss�=�pE7       �	��Zfc�A�1*

lossLT�<��V       �	&r[fc�A�1*

lossC7�;�@8Z       �	�\fc�A�1*

loss!�5=::�K       �	ʨ\fc�A�1*

loss�M<�ە       �	�E]fc�A�1*

loss���<@c��       �	��]fc�A�1*

loss�P�;��/�       �	�v^fc�A�1*

loss,.�;��ZE       �	�_fc�A�1*

loss���<c�0       �	L�_fc�A�1*

lossd�<)��       �	}?`fc�A�1*

lossnd=�Z�       �	��`fc�A�1*

loss�sg<�.�       �	 |afc�A�1*

losst��;3�أ       �	�bfc�A�1*

loss�-<<���       �	{�bfc�A�1*

lossz��<�zr�       �	[cfc�A�1*

lossW$<����       �	��cfc�A�1*

loss�1t<�`�       �	��dfc�A�1*

loss�N�=G��       �	((efc�A�1*

loss�v;���=       �	��efc�A�1*

loss\+�;��8#       �	 Sffc�A�1*

loss��<I��I       �	1�ffc�A�1*

loss,/<c��k       �	�gfc�A�1*

loss��V;�@P�       �	�hfc�A�1*

lossټ:���       �	x�hfc�A�1*

loss�b�<ɴb}       �	Mifc�A�1*

loss�M>;���       �	a�ifc�A�1*

loss��<�]߅       �	�|jfc�A�1*

loss�]�;�|p�       �	�kfc�A�1*

loss@3*=��mI       �	i�kfc�A�1*

loss$�t<�u`       �	ZElfc�A�1*

loss���<����       �	��lfc�A�1*

loss�p�;%r�       �	��mfc�A�1*

lossi&q;o�       �	�nfc�A�1*

loss�A=�"�,       �	Զnfc�A�1*

lossq�R<�OQ�       �	�Kofc�A�2*

lossL(�<1�1)       �	��ofc�A�2*

loss��Z;a�,�       �	4�pfc�A�2*

lossR��<g�Y       �	�,qfc�A�2*

lossM:���7       �	��qfc�A�2*

lossG��<F��       �	`Zrfc�A�2*

loss��:��H       �	��rfc�A�2*

lossfB=2_�-       �	7�sfc�A�2*

lossA�}<#U�       �	d!tfc�A�2*

loss�g�;!50�       �	÷tfc�A�2*

loss<z <���+       �	jKufc�A�2*

loss#��<���       �	��ufc�A�2*

loss��=魇�       �	\wvfc�A�2*

lossLsB;P~`T       �	> wfc�A�2*

lossiC=:�\       �	k�wfc�A�2*

loss΂r<�k�       �	�Txfc�A�2*

loss�� =�w�V       �	��xfc�A�2*

loss��#;a�K�       �	�~yfc�A�2*

lossM��:[_�       �	�zfc�A�2*

lossp��<���       �	p�zfc�A�2*

lossNp�;]]'       �	k{fc�A�2*

loss]X/<�.O       �	B!|fc�A�2*

loss��a;K�F       �	��|fc�A�2*

loss� :���       �	_]}fc�A�2*

loss�#<� 9�       �	A�}fc�A�2*

loss���<�G�J       �	fc�A�2*

lossD�=
�R       �	��fc�A�2*

loss� ;���J       �	�G�fc�A�2*

loss��<z�m~       �	P�fc�A�2*

loss
�;ъN�       �	8��fc�A�2*

loss��C=�``       �	�+�fc�A�2*

lossy<j���       �	�΂fc�A�2*

loss��:;-An�       �	g�fc�A�2*

loss�vf;���       �	��fc�A�2*

loss8׊;�6*       �	E��fc�A�2*

loss�!b;S�!       �	�/�fc�A�2*

loss)��<V}v�       �	 ʅfc�A�2*

loss�o�;FP Z       �	,e�fc�A�2*

loss}C�<&w߽       �	F	�fc�A�2*

loss6�:�r�)       �	��fc�A�2*

lossX��:;�B�       �	7�fc�A�2*

loss2� =��b�       �	�шfc�A�2*

loss��p=M�~       �	/n�fc�A�2*

lossܛ><�� �       �	��fc�A�2*

lossĒ�;�0��       �	Û�fc�A�2*

lossȠ<���       �	�-�fc�A�2*

loss�E<��!       �	�Ëfc�A�2*

lossa�=��C�       �	�s�fc�A�2*

loss�L<��`�       �	>$�fc�A�2*

loss:<;|P�       �	>΍fc�A�2*

loss��<�P�@       �	�u�fc�A�2*

loss�o =]H,K       �	+�fc�A�2*

loss�z�:�v��       �	콏fc�A�2*

loss�4�<V��       �	�j�fc�A�2*

loss4�<���f       �	y�fc�A�2*

loss(Vz=�I��       �	���fc�A�2*

loss��<dMe�       �	�8�fc�A�2*

loss���<��K�       �	�Ւfc�A�2*

lossΤ�<�5�       �	A��fc�A�2*

loss:G�<��r       �	��fc�A�2*

loss��a=r�N       �	�#�fc�A�2*

loss{��<��]}       �	�Εfc�A�2*

loss<�;����       �	O�fc�A�2*

loss*;���       �	I��fc�A�2*

loss&��<��E       �	��fc�A�2*

lossfv�<eBY       �	�)�fc�A�2*

loss=i�;���(       �	�a�fc�A�2*

loss�g`<?<�0       �	���fc�A�2*

loss6�:C��$       �	T��fc�A�2*

lossW�e</��       �	�(�fc�A�2*

losse�;���       �	l˜fc�A�2*

loss �=<���6       �	�j�fc�A�2*

loss�u�:��"�       �	��fc�A�2*

lossx��<"�Ee       �	���fc�A�2*

loss��]<�^�(       �	gc�fc�A�2*

loss�22: bI       �	0�fc�A�2*

loss���;^�O�       �	���fc�A�2*

loss�g958�g       �	YQ�fc�A�2*

lossv�=} BX       �	��fc�A�2*

lossƏ#;=�k]       �	닢fc�A�2*

loss���=yA��       �	w0�fc�A�2*

loss�a<c�+"       �	�ɣfc�A�2*

loss|�;��
r       �	e�fc�A�2*

loss���;�/�`       �	��fc�A�2*

loss1s8��28       �	֭�fc�A�2*

loss��K<�w��       �	-\�fc�A�2*

lossqJr;
э       �	*�fc�A�2*

loss2s�:�f6&       �	n��fc�A�2*

lossL�y<q6]$       �	�;�fc�A�2*

loss��;~�D�       �	Өfc�A�2*

lossf�U<�,�R       �	�j�fc�A�2*

loss(˪<ߎi       �	d�fc�A�2*

loss/��<*7��       �	���fc�A�2*

loss� :�4��       �	�A�fc�A�2*

losse�8=|���       �	^ګfc�A�2*

lossLL�<�}�       �	.p�fc�A�2*

loss���;���       �	��fc�A�2*

loss��<^'q�       �	���fc�A�2*

loss��<j�g�       �	7O�fc�A�2*

loss`my< c�U       �	��fc�A�2*

loss���;P��       �	͔�fc�A�2*

lossl�^<���-       �	�@�fc�A�2*

loss���:!A�f       �	�ڰfc�A�2*

lossz$�;a�x�       �	��fc�A�2*

loss*8`:��Z       �	��fc�A�2*

loss��<��:       �	U�fc�A�2*

lossHY�;�D       �	��fc�A�2*

loss�bd=p�ޝ       �	A~�fc�A�2*

loss��9���       �	w�fc�A�2*

losso@�:�.�G       �	卑fc�A�2*

lossH�<=�       �	�<�fc�A�2*

lossOv<Q�G�       �	�Ӷfc�A�2*

loss��B<Ӂ8       �	6t�fc�A�2*

lossǔ:�P       �	U�fc�A�2*

loss�/;|p&       �	K��fc�A�2*

lossD6~<���^       �	J�fc�A�2*

loss�u%;	�g�       �	c��fc�A�2*

loss��X<~��       �	ۅ�fc�A�2*

loss�.�9���       �	0*�fc�A�2*

lossM�=��X       �	�O�fc�A�2*

lossӎM<vh��       �	��fc�A�2*

losst<�bt       �	��fc�A�2*

loss�n:<���       �	���fc�A�2*

loss�ْ9�[L       �	��fc�A�2*

loss�Y:;"��7       �	K��fc�A�2*

loss��<c�H       �	kI�fc�A�2*

lossOD;mW��       �	�H�fc�A�2*

loss���<� ~�       �	��fc�A�2*

losszDb<2g��       �	^��fc�A�3*

loss��P<t���       �	*X�fc�A�3*

lossj��;3�T�       �	U��fc�A�3*

loss�g�:j���       �	J$�fc�A�3*

loss#d;/��T       �	e��fc�A�3*

loss]�</��`       �	�g�fc�A�3*

loss�9�d�       �	��fc�A�3*

lossi4�;��       �	o��fc�A�3*

loss6�(<.�*       �	�D�fc�A�3*

loss���<����       �	���fc�A�3*

lossR#<;��J1       �	;��fc�A�3*

loss\�9����       �	{-�fc�A�3*

loss];;�܀       �	x��fc�A�3*

lossA^�<��K�       �	vp�fc�A�3*

loss�j�6^4�&       �	��fc�A�3*

loss�-�:ߊ<4       �	`��fc�A�3*

lossQ�K:�tkq       �	�H�fc�A�3*

loss6"g:2i	�       �	��fc�A�3*

lossJL�;�풤       �	��fc�A�3*

lossR<:?zH�       �	�i�fc�A�3*

loss��K:7       �	C�fc�A�3*

loss�)�<��        �	���fc�A�3*

lossS;Ba��       �	�0�fc�A�3*

lossѨ<Z���       �	���fc�A�3*

lossx�;<`2+       �	sf�fc�A�3*

loss6�<��       �	���fc�A�3*

lossӁ�;i �       �	��fc�A�3*

loss:#<��5       �	&:�fc�A�3*

loss��=�Z�       �	J��fc�A�3*

loss�U,;�ܙ?       �	Hk�fc�A�3*

loss(��;ZOP�       �	:�fc�A�3*

loss`<;B3�b       �	��fc�A�3*

loss��r:H��       �	�@�fc�A�3*

loss���9�׿�       �	b��fc�A�3*

loss$+S;��j�       �	�{�fc�A�3*

loss�b�:JJ&       �	���fc�A�3*

loss�'�<8�7u       �	�:�fc�A�3*

loss̜?<n=]       �		��fc�A�3*

loss�w<����       �	�}�fc�A�3*

lossxQ�;s��       �	��fc�A�3*

lossW�<U��X       �	ܸ�fc�A�3*

loss/4�;��b       �	�P�fc�A�3*

loss�,K;n�q�       �	|��fc�A�3*

losso��;���       �	Q��fc�A�3*

loss��;J���       �	p#�fc�A�3*

loss�� ;��w2       �	=��fc�A�3*

lossz:��^�       �	zS�fc�A�3*

loss��:=y{:       �	���fc�A�3*

lossfo;��W       �	G��fc�A�3*

loss��:�A�C       �	�)�fc�A�3*

loss�x�9�V�       �	���fc�A�3*

loss�~I=4���       �	$a�fc�A�3*

losst��;��#�       �	���fc�A�3*

lossM�:[���       �	���fc�A�3*

lossH<�:�
 �       �	�$�fc�A�3*

loss}G�:��`�       �	���fc�A�3*

loss�!�;����       �	�P�fc�A�3*

lossj��:Pol�       �	���fc�A�3*

loss��<=��C       �	���fc�A�3*

loss�}�<_��       �	i;�fc�A�3*

loss�
<��8       �	��fc�A�3*

loss��:���q       �	]n�fc�A�3*

lossG9�:��n�       �	U�fc�A�3*

loss�8�<[˒�       �	Ӽ�fc�A�3*

lossaQh<�j�q       �	-`�fc�A�3*

loss��<!���       �	��fc�A�3*

loss��_;u֓m       �	��fc�A�3*

lossA;�.��       �	WB�fc�A�3*

loss��
<��       �	���fc�A�3*

loss�? <єى       �	ly�fc�A�3*

lossH<�?P=       �	i�fc�A�3*

loss��/:�R�=       �	,��fc�A�3*

loss.?G9O��=       �	P�fc�A�3*

loss���;$��t       �	�fc�A�3*

loss-=�<\$       �	-@	fc�A�3*

loss�r�<Y�'�       �	��	fc�A�3*

loss2o=�[�!       �	vn
fc�A�3*

loss��z;EX�       �	�fc�A�3*

loss�ҥ;)K�2       �	��fc�A�3*

loss�:�<�.r       �	s�fc�A�3*

loss�L�==N��       �	R(fc�A�3*

loss_��<+\�       �	�fc�A�3*

loss�HC<�&�       �	��fc�A�3*

loss��:�n}r       �	uwfc�A�3*

loss.�<�`J�       �	fc�A�3*

lossF�t;l#a       �	��fc�A�3*

lossYY<l9*b       �	�Nfc�A�3*

lossv�>-���       �	��fc�A�3*

loss��Q=�>A!       �	�fc�A�3*

lossx
�9�-ӣ       �	�Bfc�A�3*

loss,w�:Ky��       �	d�fc�A�3*

loss�J�<[1�3       �	8�fc�A�3*

lossf��<��;�       �	nLfc�A�3*

loss};���       �	��fc�A�3*

loss;<�<�M>�       �	B�fc�A�3*

loss`��:X�       �	Afc�A�3*

lossq��<�-$�       �	!fc�A�3*

losst��<��Q        �	��fc�A�3*

lossm��:�B�       �	hXfc�A�3*

loss�X�;��ś       �	4�fc�A�3*

loss-�{;��        �	I�fc�A�3*

lossZ�F=���       �	�<fc�A�3*

loss��<�<vO       �	��fc�A�3*

loss�}W;s\4�       �	��fc�A�3*

lossA?<�)s       �	�@fc�A�3*

loss�2<��ډ       �	��fc�A�3*

loss=�<<��:       �	��fc�A�3*

loss��<>kA�       �	�>fc�A�3*

loss�Ϝ<��o       �	��fc�A�3*

loss�6;��}g       �	X� fc�A�3*

loss�s<��"�       �	5!fc�A�3*

loss��=ꁃ�       �	��!fc�A�3*

loss�=My�       �	��"fc�A�3*

loss�Y;�λ�       �	�)#fc�A�3*

loss�݄<�O�       �	��#fc�A�3*

lossh�\;��i       �	ux$fc�A�3*

loss��8<����       �	\%fc�A�3*

loss.��;��S       �	��%fc�A�3*

loss��}<r�*�       �	` (fc�A�3*

loss.R�:[Э�       �	 �(fc�A�3*

loss�R=�
�"       �	J�)fc�A�3*

loss�%;��Z       �	�I*fc�A�3*

loss�@{9 ��N       �	��*fc�A�3*

loss��b;�ۮ       �	a�+fc�A�3*

loss�Ε<�ai#       �	{-,fc�A�3*

lossF�:�5"       �	5%-fc�A�3*

loss�o�=2r�       �	��-fc�A�3*

lossJ��<�$��       �	�j.fc�A�3*

loss8�9iPGa       �	�/fc�A�4*

loss=ٮ:��$l       �	�/fc�A�4*

lossl6�:��A       �	�L0fc�A�4*

loss���;�/�       �	��0fc�A�4*

loss�s=���M       �	?�1fc�A�4*

loss�t�;�X��       �	=D2fc�A�4*

loss�YF;���6       �	"�2fc�A�4*

loss (:��V>       �	2v3fc�A�4*

lossO�<O��       �	&4fc�A�4*

loss7
d<�t;^       �	��4fc�A�4*

lossƃ�;h��       �	�X5fc�A�4*

loss7��;�\=       �	��5fc�A�4*

lossr��=�m��       �	z�6fc�A�4*

lossM6�<�}��       �	�'7fc�A�4*

loss��<��	       �	�7fc�A�4*

loss���<^�H�       �	�\8fc�A�4*

loss��r<ѹ�        �	��8fc�A�4*

loss˞=�ɟ�       �	m�9fc�A�4*

loss�a�:��2W       �	�):fc�A�4*

loss�e�<n��(       �	��:fc�A�4*

lossN< �       �	�c;fc�A�4*

loss��<x�3�       �	�<fc�A�4*

lossj�;=Ђ+�       �	.=fc�A�4*

loss���;j[��       �	u�=fc�A�4*

losst:�<��bq       �	G�>fc�A�4*

loss���:���3       �	�]?fc�A�4*

lossa�)<��e�       �	 ]@fc�A�4*

loss���;~f�d       �	BAfc�A�4*

lossv��;y�J        �	��Afc�A�4*

loss�uy;M1?       �	S�Bfc�A�4*

loss��B;�.��       �	_Cfc�A�4*

loss��<�*��       �	�5Dfc�A�4*

loss]�	>.Z�       �	s�Dfc�A�4*

loss@H<�朵       �	;�Efc�A�4*

loss�S�<K��       �	�hFfc�A�4*

loss#�8<�<�       �	"Gfc�A�4*

loss��F<(��       �	v�Gfc�A�4*

loss�;����       �	vlHfc�A�4*

loss� ;;�˸       �	�Ifc�A�4*

loss� Z<�j��       �	��Ifc�A�4*

loss�]�;�Z-       �	�DJfc�A�4*

loss	k�<�|�       �	��Jfc�A�4*

loss��"=ǯ�-       �	׆Kfc�A�4*

loss�M�<�&�=       �	�/Lfc�A�4*

loss=��;�QB�       �	��Lfc�A�4*

loss�Z�;�4XA       �	�qMfc�A�4*

lossN�y:�;<       �	Nfc�A�4*

lossN	;7\ql       �	3�Nfc�A�4*

lossFA�;�T�T       �	�lOfc�A�4*

loss}��:l$�       �	�Pfc�A�4*

loss8�<ȡj�       �	��Pfc�A�4*

lossZ9�;f��I       �	�XQfc�A�4*

loss�-q<l�        �	��Qfc�A�4*

lossО�<=��^       �	1�Rfc�A�4*

loss�<)��A       �	�;Sfc�A�4*

loss�{�<�k��       �	n�Sfc�A�4*

losspZ�<���n       �	��Tfc�A�4*

lossX#�;`�:�       �	�-Ufc�A�4*

loss��;=��       �	[�Ufc�A�4*

loss؏<�r�V       �	GwVfc�A�4*

lossjI;���       �	Wfc�A�4*

lossc^�:6e��       �	��Wfc�A�4*

loss	-�;��z#       �	7pXfc�A�4*

lossD�9<�P       �	Yfc�A�4*

lossY�</wR8       �	�Yfc�A�4*

losszsB=՜�       �	|bZfc�A�4*

loss���;��        �	V[fc�A�4*

loss Դ;y
�z       �	��[fc�A�4*

loss�?=9�_       �	�a\fc�A�4*

loss�;       �	1	]fc�A�4*

lossv��;�V�2       �	��]fc�A�4*

loss��	;��       �	�U^fc�A�4*

loss�m�:���       �	�^fc�A�4*

loss�2==V9$       �	͒_fc�A�4*

loss�̊;(Q       �	4`fc�A�4*

loss1f<VZ��       �	��`fc�A�4*

lossɯ�<.�D       �	�pafc�A�4*

loss��;!�ω       �	cbfc�A�4*

loss��f<]!K�       �	]�bfc�A�4*

loss��<��       �	�>cfc�A�4*

loss�wX;p)�       �	�cfc�A�4*

loss���:Z       �	�xdfc�A�4*

loss7�a<[�[�       �	+efc�A�4*

lossQ%�:n_��       �	ͱefc�A�4*

loss�T:��
6       �	�Gffc�A�4*

loss�}< v��       �	��ffc�A�4*

loss/�;Q|"!       �	
�gfc�A�4*

loss"�<��,�       �	�$hfc�A�4*

loss@=�\D	       �	'�hfc�A�4*

losst��=.�	       �	�\ifc�A�4*

loss�1�;R��       �	]�ifc�A�4*

loss1.�<M/�       �	�jfc�A�4*

loss'@;�F�       �	'/kfc�A�4*

lossŲ�;8H=       �	�kfc�A�4*

lossT/=X���       �	Hplfc�A�4*

lossf�=��v�       �	lmfc�A�4*

loss��C<.�WG       �	��mfc�A�4*

loss4f�;���        �	(Enfc�A�4*

loss1@<��Q       �	��nfc�A�4*

loss���;Q�E>       �	-yofc�A�4*

loss��<ɋ-J       �	�pfc�A�4*

loss�S�:���       �	\�pfc�A�4*

loss7��=����       �	0Gqfc�A�4*

lossυF<O@>O       �	��qfc�A�4*

loss��;�gX�       �	�trfc�A�4*

lossC�<��       �	6�sfc�A�4*

loss�* <Y��b       �	wtfc�A�4*

loss��;��       �	�ufc�A�4*

loss_�:dPO       �	!:vfc�A�4*

loss|�<76��       �	�vfc�A�4*

losss�!:a}�j       �	o�wfc�A�4*

loss('/:f2�       �	�+xfc�A�4*

loss#��=��`       �	&�xfc�A�4*

lossN�;/�8       �	jyfc�A�4*

loss�Թ:��O;       �	xzfc�A�4*

loss� Y<W�M1       �	��zfc�A�4*

lossjV�:Џ>�       �	UK{fc�A�4*

losseh9g���       �	+M|fc�A�4*

loss��0;���Y       �	��|fc�A�4*

loss��;�	�       �	��}fc�A�4*

loss��;)�)       �	�]~fc�A�4*

loss4$=E���       �	vfc�A�4*

lossI=�=�C+�       �	��fc�A�4*

loss���<�w��       �	��fc�A�4*

lossXNG<M�'�       �	�m�fc�A�4*

loss4;�)L       �	���fc�A�4*

loss<�=�ʓ       �	Ƅfc�A�4*

loss*aL:�L߰       �	�څfc�A�4*

loss�y�:�Q��       �	�	�fc�A�5*

loss�<b;���u       �	+��fc�A�5*

loss �<���       �	�!�fc�A�5*

loss�{�<�p�y       �	��fc�A�5*

loss�C�<yK8       �	2�fc�A�5*

loss��<�>��       �	^G�fc�A�5*

loss(�,;�l<�       �	v�fc�A�5*

loss�i6=%�S�       �	�@�fc�A�5*

loss�<��J�       �	��fc�A�5*

loss��=l;       �	ɐ�fc�A�5*

loss�<�3�       �	2:�fc�A�5*

loss6�i:A��       �	��fc�A�5*

loss�c�<MCj�       �	��fc�A�5*

loss�m�;5��       �	�)�fc�A�5*

loss�[�=�8{       �	Βfc�A�5*

loss(o�;C���       �	j�fc�A�5*

loss���<�ȉ
       �	��fc�A�5*

lossk�<m>X�       �	��fc�A�5*

loss{��;�E�H       �	[[�fc�A�5*

loss���<�A�-       �	���fc�A�5*

loss��;����       �	ɏ�fc�A�5*

loss6��<�2\       �	'�fc�A�5*

loss6ز<��~       �	K�fc�A�5*

loss�<ďk_       �	w��fc�A�5*

loss�F�<u��       �	I�fc�A�5*

loss 5|<��޲       �	��fc�A�5*

loss8ۤ=?/�       �	W��fc�A�5*

loss��<ơiD       �	>@�fc�A�5*

loss�b<��       �	�ۛfc�A�5*

loss-�D;o��/       �	w�fc�A�5*

loss,��;wZ@�       �	Y�fc�A�5*

lossӝj<����       �	ޯ�fc�A�5*

lossbA=��F�       �	�P�fc�A�5*

loss�rK=p���       �	��fc�A�5*

loss1>�c�\       �	���fc�A�5*

loss��=L���       �	�:�fc�A�5*

lossٸ=����       �	�ܠfc�A�5*

lossV�;ۖ�       �	py�fc�A�5*

loss�^�<g��       �	��fc�A�5*

loss{�<��       �	Զ�fc�A�5*

loss���<�p�a       �	k��fc�A�5*

loss��N;& ��       �	�S�fc�A�5*

loss:��<c_%       �	���fc�A�5*

loss��k=q�5!       �	���fc�A�5*

loss�D\=f�N+       �	�.�fc�A�5*

loss];�P^�       �	�Ǧfc�A�5*

lossTn;����       �	�e�fc�A�5*

lossZ,<�>�       �	��fc�A�5*

loss�Xk;_y�       �	A��fc�A�5*

loss���;�b1        �	T6�fc�A�5*

loss�և<mٓ�       �	�ѩfc�A�5*

loss��<j��       �	�q�fc�A�5*

loss*,>J�K       �	��fc�A�5*

loss?'c<C��n       �	X��fc�A�5*

lossSpw=w       �	E�fc�A�5*

lossC�y<���~       �	H�fc�A�5*

loss�ߊ:_]"�       �	�}�fc�A�5*

lossey;AcQ       �	� �fc�A�5*

lossI�i;ٻ�.       �	ӽ�fc�A�5*

loss��*=>m��       �	�Z�fc�A�5*

loss��G<��Ҙ       �	��fc�A�5*

loss!��<!"<�       �	��fc�A�5*

loss�sM;� ��       �	��fc�A�5*

loss��Y<9�;�       �	uɱfc�A�5*

loss�B/;���       �	'i�fc�A�5*

loss���=q��       �	���fc�A�5*

lossc	<�Ɔ�       �	ᛳfc�A�5*

lossl%<����       �	�>�fc�A�5*

loss��6=�
��       �	�ִfc�A�5*

loss�E�<][e       �	n�fc�A�5*

loss_�;�4�       �	��fc�A�5*

loss��3;�o�3       �	��fc�A�5*

loss�T;�4E�       �	F�fc�A�5*

loss�?<�       �	v޷fc�A�5*

loss�P\;"�r       �	[{�fc�A�5*

loss�!�<�L�       �	#�fc�A�5*

loss=K�;&�       �	��fc�A�5*

lossv�?<����       �	1��fc�A�5*

loss�"�<�S       �	�O�fc�A�5*

loss��;y㟥       �	�T�fc�A�5*

lossS��<4o�       �	��fc�A�5*

loss���<��/�       �	���fc�A�5*

losss('<�!"       �	�B�fc�A�5*

loss��Y<t6�       �	`�fc�A�5*

lossv-=����       �	ؿfc�A�5*

lossM&-<Y���       �	���fc�A�5*

lossn�X;��D�       �	rn�fc�A�5*

lossOO:D�       �	A�fc�A�5*

lossD�;�xO�       �	c��fc�A�5*

loss#�|;8���       �	~R�fc�A�5*

loss��<'��G       �	���fc�A�5*

lossD1�<N�$       �	���fc�A�5*

loss�G�<F�Ni       �	�!�fc�A�5*

loss���<����       �	(��fc�A�5*

loss�j�:)x�       �	�Q�fc�A�5*

loss�3<�tM       �	���fc�A�5*

loss��=dDφ       �	$��fc�A�5*

losss��< !��       �	�3�fc�A�5*

loss��>;z"�       �	���fc�A�5*

lossCh�;�d[       �	�s�fc�A�5*

lossH��<6V�n       �	N
�fc�A�5*

loss!��<�r��       �	+��fc�A�5*

lossr,<�Pp       �	��fc�A�5*

loss��<8�C�       �	�%�fc�A�5*

lossn��<�->�       �	Y��fc�A�5*

loss���;��$s       �	9c�fc�A�5*

lossrS :��       �	��fc�A�5*

loss/�":�`�       �	Y��fc�A�5*

loss_��;!9Ui       �	F�fc�A�5*

loss�{�:
R�       �	>��fc�A�5*

loss�%_:�*��       �	i��fc�A�5*

loss1��<�y2n       �	��fc�A�5*

loss�z�<�82       �	T��fc�A�5*

lossJ�;��O!       �	]�fc�A�5*

loss&��<*Q�       �	#��fc�A�5*

loss��<��K       �	���fc�A�5*

loss���;�D{       �	{.�fc�A�5*

loss�0y;��Ӊ       �	X��fc�A�5*

lossTP�:�:<�       �	�_�fc�A�5*

loss�h;|F%�       �	/��fc�A�5*

loss���<�r��       �	q��fc�A�5*

loss�^+<]}�       �	�*�fc�A�5*

loss���<��h       �	���fc�A�5*

loss�__;e�w�       �	6Y�fc�A�5*

loss<��<�X�D       �	���fc�A�5*

lossy&�<z��)       �	S��fc�A�5*

loss�l�:�/��       �	�.�fc�A�5*

lossx�Z;z�-�       �	`��fc�A�5*

lossRŒ;�R�       �	a�fc�A�6*

lossT�=���       �	���fc�A�6*

loss��<�6
/       �	r��fc�A�6*

loss��c;)]M       �	��fc�A�6*

loss1w2<�͈       �	)��fc�A�6*

loss��:9��       �	)��fc�A�6*

loss��<b��       �	���fc�A�6*

loss�a�;K���       �	n�fc�A�6*

lossL� <��|       �	[��fc�A�6*

losscF�< �x�       �	�J�fc�A�6*

loss�Ӝ=�P��       �	���fc�A�6*

lossCV�;ӥK�       �	���fc�A�6*

lossJs�:�lU       �	$�fc�A�6*

lossq�d<	��       �	P��fc�A�6*

loss���<�x��       �	F[�fc�A�6*

loss��};�%�       �	HR�fc�A�6*

loss,�_<9��       �	���fc�A�6*

loss�Lt;�	�`       �	y��fc�A�6*

lossۉV<�9�       �	�-�fc�A�6*

lossVL=�       �	e��fc�A�6*

loss׫�< �!       �	�r�fc�A�6*

loss��:�5�       �	E�fc�A�6*

loss��<��C4       �	���fc�A�6*

loss���<]��       �	�8�fc�A�6*

loss���<(Z�       �	A��fc�A�6*

losswA=��       �	�h�fc�A�6*

loss(�<��2�       �	?��fc�A�6*

lossnr+<a��       �	���fc�A�6*

loss�9�:��Q�       �	5�fc�A�6*

loss�3�;ʭt       �	��fc�A�6*

loss��<0�       �	p|�fc�A�6*

loss�u;���3       �	��fc�A�6*

loss�$�;f�<y       �	u��fc�A�6*

lossp�==�t�       �	�C�fc�A�6*

lossn�<O�#�       �	���fc�A�6*

lossX�=;��-       �	�m�fc�A�6*

loss#��<�/��       �	��fc�A�6*

lossD�;Y       �	���fc�A�6*

loss9R#=���       �	Q0�fc�A�6*

loss�n�:�T�1       �	���fc�A�6*

lossD��:e,�       �	ka�fc�A�6*

loss�F�<���       �	!�fc�A�6*

lossJ�&=�&�       �	���fc�A�6*

loss��G<e=+       �	�7�fc�A�6*

losslX�;�]       �	���fc�A�6*

loss�b
;`�S       �	�c�fc�A�6*

loss�x�<���3       �	n��fc�A�6*

loss��';dA�       �	L��fc�A�6*

loss`�=Df?       �	� �fc�A�6*

loss��\=|��E       �	��fc�A�6*

loss�<���$       �	[@�fc�A�6*

lossJ�H<�[�       �	l��fc�A�6*

loss[+�<A=K       �	��fc�A�6*

loss��:=�̲       �	2Y�fc�A�6*

lossLGS=�>$       �	y�fc�A�6*

loss<��R�       �	q��fc�A�6*

loss�WM<�!�,       �	S� 	fc�A�6*

lossRd�;�j܀       �	 o	fc�A�6*

loss/�8<� ��       �	�	fc�A�6*

loss�?�<8�<       �	�H	fc�A�6*

loss�SZ;�Ď2       �	��	fc�A�6*

lossN�	<m��c       �	l�	fc�A�6*

lossF�<3���       �	�8	fc�A�6*

loss��	=�7�       �	��	fc�A�6*

loss|�;K�Y       �	��	fc�A�6*

losst<����       �	��	fc�A�6*

loss@�Q=j��$       �	X7		fc�A�6*

lossK�;��E       �	��		fc�A�6*

loss�#�<s �       �	��
	fc�A�6*

loss�g<}��       �	~;	fc�A�6*

lossDȐ:�mr       �	i�	fc�A�6*

loss�=���       �	#�	fc�A�6*

lossm��<[��       �	 *	fc�A�6*

lossj�
;��m�       �	��	fc�A�6*

loss��< 3��       �	�n	fc�A�6*

loss`�<G���       �	�	fc�A�6*

losso0=�ȟ�       �	`�	fc�A�6*

loss
�:*[�       �	~;	fc�A�6*

loss�%�;\s�       �	��	fc�A�6*

loss���<��[�       �	�	fc�A�6*

loss���;��E�       �	|C	fc�A�6*

loss��<g u       �	��	fc�A�6*

loss�Z<q��D       �	�|	fc�A�6*

loss�'9;	G<�       �	�D	fc�A�6*

loss��<���       �	��	fc�A�6*

losse�Y;�;F       �	��	fc�A�6*

loss[�=�0��       �	�(	fc�A�6*

lossį�:+���       �	��	fc�A�6*

loss�v�;���6       �	<P	fc�A�6*

lossI�;�GXc       �	��	fc�A�6*

loss`�6;;�D�       �	��	fc�A�6*

loss�GZ=�f�       �	L	fc�A�6*

loss��:!A��       �	\�	fc�A�6*

loss�X<�(�b       �	�D	fc�A�6*

lossI��:��6b       �	��	fc�A�6*

loss�u =Ok)�       �	��	fc�A�6*

loss�F�;��H�       �	�	fc�A�6*

loss�x�<(���       �	�� 	fc�A�6*

lossQ:;<�fl       �	N!	fc�A�6*

loss�,�;�Lu;       �	��!	fc�A�6*

loss �<7���       �	�"	fc�A�6*

losso�:槼       �	Y5#	fc�A�6*

losscVG=��U�       �	��#	fc�A�6*

lossq�;�I|       �	F~$	fc�A�6*

loss �;8��       �	�%	fc�A�6*

loss!�;�P       �	n�%	fc�A�6*

loss�[^:�e�W       �	V&	fc�A�6*

lossWZM<��       �	A�&	fc�A�6*

lossɞ<7�       �	S�'	fc�A�6*

loss�l+;��       �	/5(	fc�A�6*

loss�F�<�	       �	��(	fc�A�6*

loss=<�:�^G       �	 z)	fc�A�6*

loss���<���N       �	X*	fc�A�6*

loss\�m<�29       �	>�*	fc�A�6*

lossx+<*'       �	�o+	fc�A�6*

loss!2;�8;�       �	�,	fc�A�6*

loss�V�<�e[�       �	ݶ,	fc�A�6*

lossN8f<r�؇       �	�N-	fc�A�6*

loss,OC<Z��S       �	��-	fc�A�6*

loss�%�<�0&       �	Ɔ.	fc�A�6*

loss�Ҙ<]���       �	�$/	fc�A�6*

lossrE_;���2       �	�/	fc�A�6*

loss<6��       �	�S0	fc�A�6*

lossi5\;�c       �	#�0	fc�A�6*

lossa�o<e�	�       �	�1	fc�A�6*

loss�U�;�y@�       �	22	fc�A�6*

lossJ�==�0��       �	f�2	fc�A�6*

loss==Y;,N�B       �	��5	fc�A�6*

lossqF;1�K�       �	L�6	fc�A�7*

lossܺ�<���       �	�77	fc�A�7*

lossw�<M�<�       �	��7	fc�A�7*

loss�9�<�u/       �	߿8	fc�A�7*

loss�?<p���       �	sf9	fc�A�7*

loss�D=��       �	P:	fc�A�7*

loss��O<v�Z�       �	��:	fc�A�7*

loss���<�[FX       �	�h;	fc�A�7*

loss
3�<��C       �	�<	fc�A�7*

loss�1;�'�d       �	9(=	fc�A�7*

lossHB�<u��       �	3N>	fc�A�7*

loss��<��v       �	��>	fc�A�7*

loss���:�kq�       �	�?	fc�A�7*

loss���<1r��       �	��@	fc�A�7*

loss�3<>�        �	��A	fc�A�7*

loss8]�<9�w�       �	�)C	fc�A�7*

lossxê;_b�P       �	R�C	fc�A�7*

lossJK�;���Y       �	"�D	fc�A�7*

lossʭ�<��_#       �	�E	fc�A�7*

loss�0�<��       �	��F	fc�A�7*

loss�yv;��}�       �	k�G	fc�A�7*

loss�`�;J��       �	i�H	fc�A�7*

loss��:�4��       �	�pI	fc�A�7*

loss};��9�       �	ZGJ	fc�A�7*

loss�i=P+B�       �	��J	fc�A�7*

lossd�;.��       �	��K	fc�A�7*

loss!b�<��L       �	��L	fc�A�7*

loss�s<��_       �	�%M	fc�A�7*

loss�r$:�@��       �	�M	fc�A�7*

loss.:+U+z       �	hN	fc�A�7*

loss�#7;g�D�       �	l	O	fc�A�7*

lossD=
K�       �	o�O	fc�A�7*

lossq��:\��;       �	yWP	fc�A�7*

lossLò;�#5       �	��P	fc�A�7*

loss�<Lڙ       �	y�Q	fc�A�7*

loss�ȓ:W��_       �	4+R	fc�A�7*

loss�t�;����       �	��R	fc�A�7*

loss�;9ø       �	�iS	fc�A�7*

lossҧ<*�O�       �	�T	fc�A�7*

loss�M<� Yg       �	�T	fc�A�7*

loss�A
=E�]       �	wU	fc�A�7*

lossX�F;U��B       �	�iV	fc�A�7*

loss_R;�ߕ�       �	QW	fc�A�7*

loss�P<W+�s       �	9�W	fc�A�7*

loss��:��W�       �	�NX	fc�A�7*

loss�<��g       �	A�X	fc�A�7*

loss���;�p�       �	ȚY	fc�A�7*

loss�Q;��/C       �	,D[	fc�A�7*

loss��<��=�       �	^�[	fc�A�7*

loss3�1<}r�       �	��\	fc�A�7*

loss��3;��!�       �	dx]	fc�A�7*

lossix�<��(       �	a^	fc�A�7*

loss�<=9[       �	�^	fc�A�7*

loss�A:����       �	bN_	fc�A�7*

lossZ �:�෴       �	��_	fc�A�7*

loss�b5:����       �	g}`	fc�A�7*

loss�<d#��       �	"a	fc�A�7*

loss�8:;��       �	G�a	fc�A�7*

loss�;Z��Q       �	�ib	fc�A�7*

loss�
�;�x{�       �	�?c	fc�A�7*

loss�%�;�+       �	`�c	fc�A�7*

lossj�Y<C٠       �	y�d	fc�A�7*

loss��;M%�
       �	�Ge	fc�A�7*

loss�;-��q       �	��f	fc�A�7*

loss+!�:S"�       �	Âg	fc�A�7*

lossI!�=�p�       �	�h	fc�A�7*

lossLp
<��2�       �	��h	fc�A�7*

loss�Uw;�F[       �	1|i	fc�A�7*

lossigP: V��       �	�"j	fc�A�7*

loss_�<&ck�       �	��j	fc�A�7*

loss@�;O);�       �	�bk	fc�A�7*

loss:�;��@       �	ul	fc�A�7*

loss-Vi;��U�       �	��l	fc�A�7*

loss�)<�?��       �	�Vm	fc�A�7*

loss�p<�f�*       �	>n	fc�A�7*

loss���;��pJ       �	'�n	fc�A�7*

loss��;�@Gm       �	�@o	fc�A�7*

loss��;�?��       �	8�o	fc�A�7*

loss�0;;��Z       �	�p	fc�A�7*

loss��O;ڬ|�       �	)&q	fc�A�7*

loss�0<�cLw       �	`r	fc�A�7*

loss�
�;s|�       �	ır	fc�A�7*

loss��y;�;�       �	[^s	fc�A�7*

loss�q�:j$]�       �	�t	fc�A�7*

loss!�v;4F\       �	~�t	fc�A�7*

loss���;1�+       �	�Ru	fc�A�7*

lossۀ�:�C&�       �	�u	fc�A�7*

loss3kz;Ca>�       �	�v	fc�A�7*

loss<�<'
�       �	�Dw	fc�A�7*

lossÆP=�z�       �	B�w	fc�A�7*

loss�4�; ��n       �	"�x	fc�A�7*

loss���:��Of       �	�=y	fc�A�7*

loss_g=^&��       �	u�y	fc�A�7*

loss6�F<;2       �	s�z	fc�A�7*

loss1��7�3	n       �	%{	fc�A�7*

loss��;p��       �	�{	fc�A�7*

loss;�+9�ҏ<       �	�]|	fc�A�7*

loss1&�:�o7�       �	\U}	fc�A�7*

loss��V;��       �	�}~	fc�A�7*

lossl��8��+       �	�	fc�A�7*

loss��:T4�       �	/�	fc�A�7*

lossT^�;G��       �	_��	fc�A�7*

lossMro8eȸz       �	�v�	fc�A�7*

loss��m8��r       �	6�	fc�A�7*

loss�	�8ّ]R       �	��	fc�A�7*

lossM�<�&�&       �	�ă	fc�A�7*

lossOz
=��a       �	
f�	fc�A�7*

loss��9�o�5       �	��	fc�A�7*

loss]
:y��       �	<م	fc�A�7*

loss)<.�(:       �	3��	fc�A�7*

loss4�s9��       �	81�	fc�A�7*

lossv!�<�Sd�       �	�Ї	fc�A�7*

losshހ<�Z/       �	휈	fc�A�7*

lossS$@=R�k       �	�E�	fc�A�7*

lossN�&<�l;�       �	��	fc�A�7*

loss�dk<���?       �	%��	fc�A�7*

loss���; lC�       �	�5�	fc�A�7*

lossR�;����       �	�܋	fc�A�7*

loss�x�;4Q,y       �	��	fc�A�7*

losst�;Yw�k       �	�)�	fc�A�7*

loss��;�̤�       �	I׍	fc�A�7*

loss��Q;E�}R       �	��	fc�A�7*

lossz1�<�Ǐ�       �	1�	fc�A�7*

loss|ר:�K_U       �	�я	fc�A�7*

lossn��<�T�       �	|�	fc�A�7*

loss��L=�ڢP       �	$�	fc�A�7*

loss��;�*/       �	:̑	fc�A�7*

lossfE~;��+       �	*p�	fc�A�7*

loss�)�=��W       �	��	fc�A�8*

lossh�<y��T       �	��	fc�A�8*

losss+:�A��       �	F�	fc�A�8*

loss��e;L�-@       �	\�	fc�A�8*

loss}&�;Y}]�       �	���	fc�A�8*

loss���9j�       �	�;�	fc�A�8*

loss���=����       �	��	fc�A�8*

loss��;�J��       �	c��	fc�A�8*

losss�(;>�_       �	�;�	fc�A�8*

loss=�M;.��       �	ݘ	fc�A�8*

loss���;e<�       �	ʈ�	fc�A�8*

loss@�<���       �	�0�	fc�A�8*

loss��:H��n       �	�ך	fc�A�8*

loss:��9�f��       �	�|�	fc�A�8*

loss�:���       �	�#�	fc�A�8*

lossy{�:w!�=       �	!ʜ	fc�A�8*

loss
 �;J��P       �	�t�	fc�A�8*

loss�;@y��       �	��	fc�A�8*

lossMŝ:u���       �	�ޞ	fc�A�8*

loss1��;��       �	�z�	fc�A�8*

loss{֦<��p       �	J)�	fc�A�8*

loss�f�;��pY       �	Ǆ�	fc�A�8*

loss�W;q��       �	Y4�	fc�A�8*

loss�x�<�gf�       �	sۢ	fc�A�8*

loss��;.Ff       �	B��	fc�A�8*

loss�\<(�&�       �	�L�	fc�A�8*

loss��:�3�<       �	��	fc�A�8*

lossƿ	<HS��       �	Ԟ�	fc�A�8*

loss�q=��       �	���	fc�A�8*

loss�};Q���       �	:�	fc�A�8*

lossq�+<^�f�       �	�	fc�A�8*

loss�+b:{;Ȥ       �	��	fc�A�8*

loss��<>뵍       �	�,�	fc�A�8*

losshє;�C�/