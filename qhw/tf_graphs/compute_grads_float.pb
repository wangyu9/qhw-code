
;
VPlaceholder*
dtype0*
shape:?????????
;
FPlaceholder*
dtype0*
shape:?????????
D
WPlaceholder*%
shape:??????????????????*
dtype0
8
ExpandDims/dimConst*
dtype0*
value	B : 
@

ExpandDims
ExpandDimsVExpandDims/dim*

Tdim0*
T0
:
ExpandDims_1/dimConst*
dtype0*
value	B : 
D
ExpandDims_1
ExpandDimsFExpandDims_1/dim*
T0*

Tdim0
:
ExpandDims_2/dimConst*
dtype0*
value	B : 
D
ExpandDims_2
ExpandDimsWExpandDims_2/dim*
T0*

Tdim0
J
Sum/reduction_indicesConst*
dtype0*
valueB"       
S
SumSum
ExpandDimsSum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
L
strided_slice/stackConst*!
valueB"            *
dtype0
N
strided_slice/stack_1Const*
dtype0*!
valueB"           
N
strided_slice/stack_2Const*!
valueB"         *
dtype0
?
strided_sliceStridedSlice
ExpandDimsstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
N
strided_slice_1/stackConst*!
valueB"            *
dtype0
P
strided_slice_1/stack_1Const*!
valueB"          *
dtype0
P
strided_slice_1/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_1StridedSliceExpandDims_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
7
GatherV2/axisConst*
dtype0*
value	B : 
?
GatherV2GatherV2strided_slicestrided_slice_1GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
N
strided_slice_2/stackConst*!
valueB"            *
dtype0
P
strided_slice_2/stack_1Const*!
valueB"           *
dtype0
P
strided_slice_2/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_2StridedSlice
ExpandDimsstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
N
strided_slice_3/stackConst*!
valueB"           *
dtype0
P
strided_slice_3/stack_1Const*!
valueB"          *
dtype0
P
strided_slice_3/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_3StridedSliceExpandDims_1strided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
9
GatherV2_1/axisConst*
value	B : *
dtype0
?

GatherV2_1GatherV2strided_slice_2strided_slice_3GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
N
strided_slice_4/stackConst*
dtype0*!
valueB"            
P
strided_slice_4/stack_1Const*!
valueB"           *
dtype0
P
strided_slice_4/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_4StridedSlice
ExpandDimsstrided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
N
strided_slice_5/stackConst*!
valueB"           *
dtype0
P
strided_slice_5/stack_1Const*!
valueB"          *
dtype0
P
strided_slice_5/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_5StridedSliceExpandDims_1strided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
9
GatherV2_2/axisConst*
dtype0*
value	B : 
?

GatherV2_2GatherV2strided_slice_4strided_slice_5GatherV2_2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
N
strided_slice_6/stackConst*!
valueB"            *
dtype0
P
strided_slice_6/stack_1Const*!
valueB"           *
dtype0
P
strided_slice_6/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_6StridedSlice
ExpandDimsstrided_slice_6/stackstrided_slice_6/stack_1strided_slice_6/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
N
strided_slice_7/stackConst*!
valueB"           *
dtype0
P
strided_slice_7/stack_1Const*!
valueB"          *
dtype0
P
strided_slice_7/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_7StridedSliceExpandDims_1strided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
9
GatherV2_3/axisConst*
dtype0*
value	B : 
?

GatherV2_3GatherV2strided_slice_6strided_slice_7GatherV2_3/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0
Y
stackPackGatherV2
GatherV2_1
GatherV2_2
GatherV2_3*
T0*

axis*
N
4
stack_1Packstack*
T0*

axis *
N
R
strided_slice_8/stackConst*%
valueB"                *
dtype0
T
strided_slice_8/stack_1Const*%
valueB"              *
dtype0
T
strided_slice_8/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_8StridedSlicestack_1strided_slice_8/stackstrided_slice_8/stack_1strided_slice_8/stack_2*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
Index0*
T0
R
strided_slice_9/stackConst*%
valueB"               *
dtype0
T
strided_slice_9/stack_1Const*
dtype0*%
valueB"              
T
strided_slice_9/stack_2Const*
dtype0*%
valueB"            
?
strided_slice_9StridedSlicestack_1strided_slice_9/stackstrided_slice_9/stack_1strided_slice_9/stack_2*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
Index0*
T0
S
strided_slice_10/stackConst*
dtype0*%
valueB"               
U
strided_slice_10/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_10/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_10StridedSlicestack_1strided_slice_10/stackstrided_slice_10/stack_1strided_slice_10/stack_2*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
Index0*
T0
S
strided_slice_11/stackConst*%
valueB"               *
dtype0
U
strided_slice_11/stack_1Const*
dtype0*%
valueB"              
U
strided_slice_11/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_11StridedSlicestack_1strided_slice_11/stackstrided_slice_11/stack_1strided_slice_11/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask

6
subSubstrided_slice_11strided_slice_8*
T0
8
sub_1Substrided_slice_10strided_slice_8*
T0
7
sub_2Substrided_slice_9strided_slice_8*
T0
8
sub_3Substrided_slice_11strided_slice_9*
T0
8
sub_4Substrided_slice_10strided_slice_9*
T0
%
CrossCrosssub_4sub_3*
T0
&
norm/mulMulCrossCross*
T0
H
norm/Sum/reduction_indicesConst*
valueB:*
dtype0
[
norm/SumSumnorm/mulnorm/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims(
$
	norm/SqrtSqrtnorm/Sum*
T0
B
norm/SqueezeSqueeze	norm/Sqrt*
squeeze_dims
*
T0
2
div/xConst*
valueB
 *  ??*
dtype0
,
divRealDivdiv/xnorm/Squeeze*
T0
C
ExpandDims_3/dimConst*
valueB :
?????????*
dtype0
F
ExpandDims_3
ExpandDimsdivExpandDims_3/dim*

Tdim0*
T0
(
mulMulCrossExpandDims_3*
T0
!
mul_1Mulmulsub_2*
T0
A
Sum_1/reduction_indicesConst*
value	B :*
dtype0
R
Sum_1Summul_1Sum_1/reduction_indices*

Tidx0*
	keep_dims( *
T0
4
div_1/xConst*
valueB
 *  ??*
dtype0
)
div_1RealDivdiv_1/xSum_1*
T0
C
ExpandDims_4/dimConst*
valueB :
?????????*
dtype0
H
ExpandDims_4
ExpandDimsdiv_1ExpandDims_4/dim*

Tdim0*
T0
(
mul_2MulmulExpandDims_4*
T0
O
strided_slice_12/stackConst*!
valueB"            *
dtype0
Q
strided_slice_12/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_12/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_12StridedSlice
ExpandDimsstrided_slice_12/stackstrided_slice_12/stack_1strided_slice_12/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
O
strided_slice_13/stackConst*!
valueB"            *
dtype0
Q
strided_slice_13/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_13/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_13StridedSliceExpandDims_1strided_slice_13/stackstrided_slice_13/stack_1strided_slice_13/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
9
GatherV2_4/axisConst*
value	B : *
dtype0
?

GatherV2_4GatherV2strided_slice_12strided_slice_13GatherV2_4/axis*

batch_dims *
Tindices0*
Tparams0*
Taxis0
O
strided_slice_14/stackConst*
dtype0*!
valueB"            
Q
strided_slice_14/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_14/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_14StridedSlice
ExpandDimsstrided_slice_14/stackstrided_slice_14/stack_1strided_slice_14/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
shrink_axis_mask
O
strided_slice_15/stackConst*!
valueB"           *
dtype0
Q
strided_slice_15/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_15/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_15StridedSliceExpandDims_1strided_slice_15/stackstrided_slice_15/stack_1strided_slice_15/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask
9
GatherV2_5/axisConst*
value	B : *
dtype0
?

GatherV2_5GatherV2strided_slice_14strided_slice_15GatherV2_5/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
O
strided_slice_16/stackConst*!
valueB"            *
dtype0
Q
strided_slice_16/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_16/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_16StridedSlice
ExpandDimsstrided_slice_16/stackstrided_slice_16/stack_1strided_slice_16/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
O
strided_slice_17/stackConst*!
valueB"           *
dtype0
Q
strided_slice_17/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_17/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_17StridedSliceExpandDims_1strided_slice_17/stackstrided_slice_17/stack_1strided_slice_17/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
shrink_axis_mask
9
GatherV2_6/axisConst*
value	B : *
dtype0
?

GatherV2_6GatherV2strided_slice_16strided_slice_17GatherV2_6/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
O
strided_slice_18/stackConst*!
valueB"            *
dtype0
Q
strided_slice_18/stack_1Const*
dtype0*!
valueB"           
Q
strided_slice_18/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_18StridedSlice
ExpandDimsstrided_slice_18/stackstrided_slice_18/stack_1strided_slice_18/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
O
strided_slice_19/stackConst*
dtype0*!
valueB"           
Q
strided_slice_19/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_19/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_19StridedSliceExpandDims_1strided_slice_19/stackstrided_slice_19/stack_1strided_slice_19/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
9
GatherV2_7/axisConst*
value	B : *
dtype0
?

GatherV2_7GatherV2strided_slice_18strided_slice_19GatherV2_7/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
]
stack_2Pack
GatherV2_4
GatherV2_5
GatherV2_6
GatherV2_7*
T0*

axis*
N
6
stack_3Packstack_2*
T0*

axis *
N
S
strided_slice_20/stackConst*%
valueB"               *
dtype0
U
strided_slice_20/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_20/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_20StridedSlicestack_3strided_slice_20/stackstrided_slice_20/stack_1strided_slice_20/stack_2*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
Index0*
T0*
shrink_axis_mask
S
strided_slice_21/stackConst*%
valueB"               *
dtype0
U
strided_slice_21/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_21/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_21StridedSlicestack_3strided_slice_21/stackstrided_slice_21/stack_1strided_slice_21/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask

S
strided_slice_22/stackConst*%
valueB"               *
dtype0
U
strided_slice_22/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_22/stack_2Const*
dtype0*%
valueB"            
?
strided_slice_22StridedSlicestack_3strided_slice_22/stackstrided_slice_22/stack_1strided_slice_22/stack_2*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
Index0*
T0
S
strided_slice_23/stackConst*%
valueB"                *
dtype0
U
strided_slice_23/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_23/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_23StridedSlicestack_3strided_slice_23/stackstrided_slice_23/stack_1strided_slice_23/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
*
new_axis_mask *
end_mask

9
sub_5Substrided_slice_23strided_slice_20*
T0
9
sub_6Substrided_slice_22strided_slice_20*
T0
9
sub_7Substrided_slice_21strided_slice_20*
T0
9
sub_8Substrided_slice_23strided_slice_21*
T0
9
sub_9Substrided_slice_22strided_slice_21*
T0
'
Cross_1Crosssub_9sub_8*
T0
,

norm_1/mulMulCross_1Cross_1*
T0
J
norm_1/Sum/reduction_indicesConst*
dtype0*
valueB:
a

norm_1/SumSum
norm_1/mulnorm_1/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims(
(
norm_1/SqrtSqrt
norm_1/Sum*
T0
F
norm_1/SqueezeSqueezenorm_1/Sqrt*
squeeze_dims
*
T0
4
div_2/xConst*
valueB
 *  ??*
dtype0
2
div_2RealDivdiv_2/xnorm_1/Squeeze*
T0
C
ExpandDims_5/dimConst*
dtype0*
valueB :
?????????
H
ExpandDims_5
ExpandDimsdiv_2ExpandDims_5/dim*
T0*

Tdim0
,
mul_3MulCross_1ExpandDims_5*
T0
#
mul_4Mulmul_3sub_7*
T0
A
Sum_2/reduction_indicesConst*
value	B :*
dtype0
R
Sum_2Summul_4Sum_2/reduction_indices*

Tidx0*
	keep_dims( *
T0
4
div_3/xConst*
valueB
 *  ??*
dtype0
)
div_3RealDivdiv_3/xSum_2*
T0
C
ExpandDims_6/dimConst*
dtype0*
valueB :
?????????
H
ExpandDims_6
ExpandDimsdiv_3ExpandDims_6/dim*

Tdim0*
T0
*
mul_5Mulmul_3ExpandDims_6*
T0
O
strided_slice_24/stackConst*!
valueB"            *
dtype0
Q
strided_slice_24/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_24/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_24StridedSlice
ExpandDimsstrided_slice_24/stackstrided_slice_24/stack_1strided_slice_24/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
O
strided_slice_25/stackConst*!
valueB"            *
dtype0
Q
strided_slice_25/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_25/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_25StridedSliceExpandDims_1strided_slice_25/stackstrided_slice_25/stack_1strided_slice_25/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
9
GatherV2_8/axisConst*
value	B : *
dtype0
?

GatherV2_8GatherV2strided_slice_24strided_slice_25GatherV2_8/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
O
strided_slice_26/stackConst*
dtype0*!
valueB"            
Q
strided_slice_26/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_26/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_26StridedSlice
ExpandDimsstrided_slice_26/stackstrided_slice_26/stack_1strided_slice_26/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
O
strided_slice_27/stackConst*!
valueB"           *
dtype0
Q
strided_slice_27/stack_1Const*
dtype0*!
valueB"          
Q
strided_slice_27/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_27StridedSliceExpandDims_1strided_slice_27/stackstrided_slice_27/stack_1strided_slice_27/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
9
GatherV2_9/axisConst*
dtype0*
value	B : 
?

GatherV2_9GatherV2strided_slice_26strided_slice_27GatherV2_9/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
O
strided_slice_28/stackConst*!
valueB"            *
dtype0
Q
strided_slice_28/stack_1Const*
dtype0*!
valueB"           
Q
strided_slice_28/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_28StridedSlice
ExpandDimsstrided_slice_28/stackstrided_slice_28/stack_1strided_slice_28/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
O
strided_slice_29/stackConst*
dtype0*!
valueB"           
Q
strided_slice_29/stack_1Const*
dtype0*!
valueB"          
Q
strided_slice_29/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_29StridedSliceExpandDims_1strided_slice_29/stackstrided_slice_29/stack_1strided_slice_29/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
:
GatherV2_10/axisConst*
value	B : *
dtype0
?
GatherV2_10GatherV2strided_slice_28strided_slice_29GatherV2_10/axis*

batch_dims *
Tindices0*
Tparams0*
Taxis0
O
strided_slice_30/stackConst*
dtype0*!
valueB"            
Q
strided_slice_30/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_30/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_30StridedSlice
ExpandDimsstrided_slice_30/stackstrided_slice_30/stack_1strided_slice_30/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
O
strided_slice_31/stackConst*!
valueB"           *
dtype0
Q
strided_slice_31/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_31/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_31StridedSliceExpandDims_1strided_slice_31/stackstrided_slice_31/stack_1strided_slice_31/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
:
GatherV2_11/axisConst*
dtype0*
value	B : 
?
GatherV2_11GatherV2strided_slice_30strided_slice_31GatherV2_11/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
_
stack_4Pack
GatherV2_8
GatherV2_9GatherV2_10GatherV2_11*
T0*

axis*
N
6
stack_5Packstack_4*
T0*

axis *
N
S
strided_slice_32/stackConst*%
valueB"               *
dtype0
U
strided_slice_32/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_32/stack_2Const*
dtype0*%
valueB"            
?
strided_slice_32StridedSlicestack_5strided_slice_32/stackstrided_slice_32/stack_1strided_slice_32/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
*
new_axis_mask *
end_mask

S
strided_slice_33/stackConst*
dtype0*%
valueB"               
U
strided_slice_33/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_33/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_33StridedSlicestack_5strided_slice_33/stackstrided_slice_33/stack_1strided_slice_33/stack_2*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
T0*
Index0
S
strided_slice_34/stackConst*
dtype0*%
valueB"                
U
strided_slice_34/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_34/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_34StridedSlicestack_5strided_slice_34/stackstrided_slice_34/stack_1strided_slice_34/stack_2*
end_mask
*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
*
new_axis_mask 
S
strided_slice_35/stackConst*%
valueB"               *
dtype0
U
strided_slice_35/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_35/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_35StridedSlicestack_5strided_slice_35/stackstrided_slice_35/stack_1strided_slice_35/stack_2*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
Index0*
T0*
shrink_axis_mask
:
sub_10Substrided_slice_35strided_slice_32*
T0
:
sub_11Substrided_slice_34strided_slice_32*
T0
:
sub_12Substrided_slice_33strided_slice_32*
T0
:
sub_13Substrided_slice_35strided_slice_33*
T0
:
sub_14Substrided_slice_34strided_slice_33*
T0
)
Cross_2Crosssub_14sub_13*
T0
,

norm_2/mulMulCross_2Cross_2*
T0
J
norm_2/Sum/reduction_indicesConst*
valueB:*
dtype0
a

norm_2/SumSum
norm_2/mulnorm_2/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
(
norm_2/SqrtSqrt
norm_2/Sum*
T0
F
norm_2/SqueezeSqueezenorm_2/Sqrt*
squeeze_dims
*
T0
4
div_4/xConst*
dtype0*
valueB
 *  ??
2
div_4RealDivdiv_4/xnorm_2/Squeeze*
T0
C
ExpandDims_7/dimConst*
valueB :
?????????*
dtype0
H
ExpandDims_7
ExpandDimsdiv_4ExpandDims_7/dim*

Tdim0*
T0
,
mul_6MulCross_2ExpandDims_7*
T0
$
mul_7Mulmul_6sub_12*
T0
A
Sum_3/reduction_indicesConst*
dtype0*
value	B :
R
Sum_3Summul_7Sum_3/reduction_indices*

Tidx0*
	keep_dims( *
T0
4
div_5/xConst*
valueB
 *  ??*
dtype0
)
div_5RealDivdiv_5/xSum_3*
T0
C
ExpandDims_8/dimConst*
valueB :
?????????*
dtype0
H
ExpandDims_8
ExpandDimsdiv_5ExpandDims_8/dim*
T0*

Tdim0
*
mul_8Mulmul_6ExpandDims_8*
T0
O
strided_slice_36/stackConst*!
valueB"            *
dtype0
Q
strided_slice_36/stack_1Const*
dtype0*!
valueB"           
Q
strided_slice_36/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_36StridedSlice
ExpandDimsstrided_slice_36/stackstrided_slice_36/stack_1strided_slice_36/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
O
strided_slice_37/stackConst*
dtype0*!
valueB"            
Q
strided_slice_37/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_37/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_37StridedSliceExpandDims_1strided_slice_37/stackstrided_slice_37/stack_1strided_slice_37/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
:
GatherV2_12/axisConst*
value	B : *
dtype0
?
GatherV2_12GatherV2strided_slice_36strided_slice_37GatherV2_12/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
O
strided_slice_38/stackConst*
dtype0*!
valueB"            
Q
strided_slice_38/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_38/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_38StridedSlice
ExpandDimsstrided_slice_38/stackstrided_slice_38/stack_1strided_slice_38/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
O
strided_slice_39/stackConst*!
valueB"           *
dtype0
Q
strided_slice_39/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_39/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_39StridedSliceExpandDims_1strided_slice_39/stackstrided_slice_39/stack_1strided_slice_39/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
:
GatherV2_13/axisConst*
value	B : *
dtype0
?
GatherV2_13GatherV2strided_slice_38strided_slice_39GatherV2_13/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
O
strided_slice_40/stackConst*!
valueB"            *
dtype0
Q
strided_slice_40/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_40/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_40StridedSlice
ExpandDimsstrided_slice_40/stackstrided_slice_40/stack_1strided_slice_40/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
O
strided_slice_41/stackConst*!
valueB"           *
dtype0
Q
strided_slice_41/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_41/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_41StridedSliceExpandDims_1strided_slice_41/stackstrided_slice_41/stack_1strided_slice_41/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
:
GatherV2_14/axisConst*
value	B : *
dtype0
?
GatherV2_14GatherV2strided_slice_40strided_slice_41GatherV2_14/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
O
strided_slice_42/stackConst*!
valueB"            *
dtype0
Q
strided_slice_42/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_42/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_42StridedSlice
ExpandDimsstrided_slice_42/stackstrided_slice_42/stack_1strided_slice_42/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
O
strided_slice_43/stackConst*!
valueB"           *
dtype0
Q
strided_slice_43/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_43/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_43StridedSliceExpandDims_1strided_slice_43/stackstrided_slice_43/stack_1strided_slice_43/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
:
GatherV2_15/axisConst*
value	B : *
dtype0
?
GatherV2_15GatherV2strided_slice_42strided_slice_43GatherV2_15/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0
a
stack_6PackGatherV2_12GatherV2_13GatherV2_14GatherV2_15*
N*
T0*

axis
6
stack_7Packstack_6*
T0*

axis *
N
S
strided_slice_44/stackConst*%
valueB"               *
dtype0
U
strided_slice_44/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_44/stack_2Const*
dtype0*%
valueB"            
?
strided_slice_44StridedSlicestack_7strided_slice_44/stackstrided_slice_44/stack_1strided_slice_44/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask

S
strided_slice_45/stackConst*
dtype0*%
valueB"                
U
strided_slice_45/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_45/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_45StridedSlicestack_7strided_slice_45/stackstrided_slice_45/stack_1strided_slice_45/stack_2*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
T0*
Index0
S
strided_slice_46/stackConst*%
valueB"               *
dtype0
U
strided_slice_46/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_46/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_46StridedSlicestack_7strided_slice_46/stackstrided_slice_46/stack_1strided_slice_46/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask

S
strided_slice_47/stackConst*%
valueB"               *
dtype0
U
strided_slice_47/stack_1Const*
dtype0*%
valueB"              
U
strided_slice_47/stack_2Const*
dtype0*%
valueB"            
?
strided_slice_47StridedSlicestack_7strided_slice_47/stackstrided_slice_47/stack_1strided_slice_47/stack_2*
end_mask
*
Index0*
T0*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask 
:
sub_15Substrided_slice_47strided_slice_44*
T0
:
sub_16Substrided_slice_46strided_slice_44*
T0
:
sub_17Substrided_slice_45strided_slice_44*
T0
:
sub_18Substrided_slice_47strided_slice_45*
T0
:
sub_19Substrided_slice_46strided_slice_45*
T0
)
Cross_3Crosssub_19sub_18*
T0
,

norm_3/mulMulCross_3Cross_3*
T0
J
norm_3/Sum/reduction_indicesConst*
valueB:*
dtype0
a

norm_3/SumSum
norm_3/mulnorm_3/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims(
(
norm_3/SqrtSqrt
norm_3/Sum*
T0
F
norm_3/SqueezeSqueezenorm_3/Sqrt*
squeeze_dims
*
T0
4
div_6/xConst*
dtype0*
valueB
 *  ??
2
div_6RealDivdiv_6/xnorm_3/Squeeze*
T0
C
ExpandDims_9/dimConst*
valueB :
?????????*
dtype0
H
ExpandDims_9
ExpandDimsdiv_6ExpandDims_9/dim*

Tdim0*
T0
,
mul_9MulCross_3ExpandDims_9*
T0
%
mul_10Mulmul_9sub_17*
T0
A
Sum_4/reduction_indicesConst*
value	B :*
dtype0
S
Sum_4Summul_10Sum_4/reduction_indices*

Tidx0*
	keep_dims( *
T0
4
div_7/xConst*
valueB
 *  ??*
dtype0
)
div_7RealDivdiv_7/xSum_4*
T0
D
ExpandDims_10/dimConst*
valueB :
?????????*
dtype0
J
ExpandDims_10
ExpandDimsdiv_7ExpandDims_10/dim*

Tdim0*
T0
,
mul_11Mulmul_9ExpandDims_10*
T0
D
strided_slice_48/stackConst*
valueB: *
dtype0
F
strided_slice_48/stack_1Const*
valueB:*
dtype0
F
strided_slice_48/stack_2Const*
valueB:*
dtype0
?
strided_slice_48StridedSliceExpandDims_2strided_slice_48/stackstrided_slice_48/stack_1strided_slice_48/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
D
strided_slice_49/stackConst*
valueB: *
dtype0
F
strided_slice_49/stack_1Const*
valueB:*
dtype0
F
strided_slice_49/stack_2Const*
valueB:*
dtype0
?
strided_slice_49StridedSliceExpandDims_1strided_slice_49/stackstrided_slice_49/stack_1strided_slice_49/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
K
strided_slice_50/stackConst*
valueB"        *
dtype0
M
strided_slice_50/stack_1Const*
valueB"       *
dtype0
M
strided_slice_50/stack_2Const*
valueB"      *
dtype0
?
strided_slice_50StridedSlicestrided_slice_49strided_slice_50/stackstrided_slice_50/stack_1strided_slice_50/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
:
GatherV2_16/axisConst*
value	B : *
dtype0
?
GatherV2_16GatherV2strided_slice_48strided_slice_50GatherV2_16/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
K
strided_slice_51/stackConst*
valueB"        *
dtype0
M
strided_slice_51/stack_1Const*
valueB"       *
dtype0
M
strided_slice_51/stack_2Const*
valueB"      *
dtype0
?
strided_slice_51StridedSlicemul_2strided_slice_51/stackstrided_slice_51/stack_1strided_slice_51/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
D
ExpandDims_11/dimConst*
dtype0*
valueB :
?????????
U
ExpandDims_11
ExpandDimsstrided_slice_51ExpandDims_11/dim*
T0*

Tdim0
K
strided_slice_52/stackConst*
valueB"        *
dtype0
M
strided_slice_52/stack_1Const*
valueB"       *
dtype0
M
strided_slice_52/stack_2Const*
valueB"      *
dtype0
?
strided_slice_52StridedSlicestrided_slice_49strided_slice_52/stackstrided_slice_52/stack_1strided_slice_52/stack_2*
end_mask*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
:
GatherV2_17/axisConst*
dtype0*
value	B : 
?
GatherV2_17GatherV2strided_slice_48strided_slice_52GatherV2_17/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
2
mul_12MulExpandDims_11GatherV2_17*
T0
K
strided_slice_53/stackConst*
valueB"        *
dtype0
M
strided_slice_53/stack_1Const*
valueB"       *
dtype0
M
strided_slice_53/stack_2Const*
valueB"      *
dtype0
?
strided_slice_53StridedSlicemul_5strided_slice_53/stackstrided_slice_53/stack_1strided_slice_53/stack_2*
end_mask*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
D
ExpandDims_12/dimConst*
valueB :
?????????*
dtype0
U
ExpandDims_12
ExpandDimsstrided_slice_53ExpandDims_12/dim*
T0*

Tdim0
K
strided_slice_54/stackConst*
dtype0*
valueB"       
M
strided_slice_54/stack_1Const*
valueB"       *
dtype0
M
strided_slice_54/stack_2Const*
dtype0*
valueB"      
?
strided_slice_54StridedSlicestrided_slice_49strided_slice_54/stackstrided_slice_54/stack_1strided_slice_54/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
:
GatherV2_18/axisConst*
value	B : *
dtype0
?
GatherV2_18GatherV2strided_slice_48strided_slice_54GatherV2_18/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0
2
mul_13MulExpandDims_12GatherV2_18*
T0
%
addAddV2mul_12mul_13*
T0
K
strided_slice_55/stackConst*
valueB"        *
dtype0
M
strided_slice_55/stack_1Const*
dtype0*
valueB"       
M
strided_slice_55/stack_2Const*
valueB"      *
dtype0
?
strided_slice_55StridedSlicemul_8strided_slice_55/stackstrided_slice_55/stack_1strided_slice_55/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0*
shrink_axis_mask
D
ExpandDims_13/dimConst*
dtype0*
valueB :
?????????
U
ExpandDims_13
ExpandDimsstrided_slice_55ExpandDims_13/dim*

Tdim0*
T0
K
strided_slice_56/stackConst*
valueB"       *
dtype0
M
strided_slice_56/stack_1Const*
valueB"       *
dtype0
M
strided_slice_56/stack_2Const*
valueB"      *
dtype0
?
strided_slice_56StridedSlicestrided_slice_49strided_slice_56/stackstrided_slice_56/stack_1strided_slice_56/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
:
GatherV2_19/axisConst*
value	B : *
dtype0
?
GatherV2_19GatherV2strided_slice_48strided_slice_56GatherV2_19/axis*

batch_dims *
Tindices0*
Tparams0*
Taxis0
2
mul_14MulExpandDims_13GatherV2_19*
T0
$
add_1AddV2addmul_14*
T0
K
strided_slice_57/stackConst*
dtype0*
valueB"        
M
strided_slice_57/stack_1Const*
valueB"       *
dtype0
M
strided_slice_57/stack_2Const*
valueB"      *
dtype0
?
strided_slice_57StridedSlicemul_11strided_slice_57/stackstrided_slice_57/stack_1strided_slice_57/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
D
ExpandDims_14/dimConst*
valueB :
?????????*
dtype0
U
ExpandDims_14
ExpandDimsstrided_slice_57ExpandDims_14/dim*

Tdim0*
T0
K
strided_slice_58/stackConst*
valueB"       *
dtype0
M
strided_slice_58/stack_1Const*
valueB"       *
dtype0
M
strided_slice_58/stack_2Const*
valueB"      *
dtype0
?
strided_slice_58StridedSlicestrided_slice_49strided_slice_58/stackstrided_slice_58/stack_1strided_slice_58/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
:
GatherV2_20/axisConst*
value	B : *
dtype0
?
GatherV2_20GatherV2strided_slice_48strided_slice_58GatherV2_20/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
2
mul_15MulExpandDims_14GatherV2_20*
T0
&
add_2AddV2add_1mul_15*
T0

GWxIdentityadd_2*
T0
K
strided_slice_59/stackConst*
dtype0*
valueB"       
M
strided_slice_59/stack_1Const*
valueB"       *
dtype0
M
strided_slice_59/stack_2Const*
valueB"      *
dtype0
?
strided_slice_59StridedSlicemul_2strided_slice_59/stackstrided_slice_59/stack_1strided_slice_59/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
D
ExpandDims_15/dimConst*
valueB :
?????????*
dtype0
U
ExpandDims_15
ExpandDimsstrided_slice_59ExpandDims_15/dim*

Tdim0*
T0
K
strided_slice_60/stackConst*
valueB"        *
dtype0
M
strided_slice_60/stack_1Const*
valueB"       *
dtype0
M
strided_slice_60/stack_2Const*
valueB"      *
dtype0
?
strided_slice_60StridedSlicestrided_slice_49strided_slice_60/stackstrided_slice_60/stack_1strided_slice_60/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
:
GatherV2_21/axisConst*
value	B : *
dtype0
?
GatherV2_21GatherV2strided_slice_48strided_slice_60GatherV2_21/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
2
mul_16MulExpandDims_15GatherV2_21*
T0
K
strided_slice_61/stackConst*
dtype0*
valueB"       
M
strided_slice_61/stack_1Const*
valueB"       *
dtype0
M
strided_slice_61/stack_2Const*
valueB"      *
dtype0
?
strided_slice_61StridedSlicemul_5strided_slice_61/stackstrided_slice_61/stack_1strided_slice_61/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
D
ExpandDims_16/dimConst*
dtype0*
valueB :
?????????
U
ExpandDims_16
ExpandDimsstrided_slice_61ExpandDims_16/dim*

Tdim0*
T0
K
strided_slice_62/stackConst*
valueB"       *
dtype0
M
strided_slice_62/stack_1Const*
dtype0*
valueB"       
M
strided_slice_62/stack_2Const*
dtype0*
valueB"      
?
strided_slice_62StridedSlicestrided_slice_49strided_slice_62/stackstrided_slice_62/stack_1strided_slice_62/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
:
GatherV2_22/axisConst*
dtype0*
value	B : 
?
GatherV2_22GatherV2strided_slice_48strided_slice_62GatherV2_22/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
2
mul_17MulExpandDims_16GatherV2_22*
T0
'
add_3AddV2mul_16mul_17*
T0
K
strided_slice_63/stackConst*
valueB"       *
dtype0
M
strided_slice_63/stack_1Const*
valueB"       *
dtype0
M
strided_slice_63/stack_2Const*
valueB"      *
dtype0
?
strided_slice_63StridedSlicemul_8strided_slice_63/stackstrided_slice_63/stack_1strided_slice_63/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask
D
ExpandDims_17/dimConst*
dtype0*
valueB :
?????????
U
ExpandDims_17
ExpandDimsstrided_slice_63ExpandDims_17/dim*

Tdim0*
T0
K
strided_slice_64/stackConst*
valueB"       *
dtype0
M
strided_slice_64/stack_1Const*
dtype0*
valueB"       
M
strided_slice_64/stack_2Const*
valueB"      *
dtype0
?
strided_slice_64StridedSlicestrided_slice_49strided_slice_64/stackstrided_slice_64/stack_1strided_slice_64/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
:
GatherV2_23/axisConst*
value	B : *
dtype0
?
GatherV2_23GatherV2strided_slice_48strided_slice_64GatherV2_23/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
2
mul_18MulExpandDims_17GatherV2_23*
T0
&
add_4AddV2add_3mul_18*
T0
K
strided_slice_65/stackConst*
dtype0*
valueB"       
M
strided_slice_65/stack_1Const*
valueB"       *
dtype0
M
strided_slice_65/stack_2Const*
valueB"      *
dtype0
?
strided_slice_65StridedSlicemul_11strided_slice_65/stackstrided_slice_65/stack_1strided_slice_65/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
D
ExpandDims_18/dimConst*
valueB :
?????????*
dtype0
U
ExpandDims_18
ExpandDimsstrided_slice_65ExpandDims_18/dim*

Tdim0*
T0
K
strided_slice_66/stackConst*
valueB"       *
dtype0
M
strided_slice_66/stack_1Const*
valueB"       *
dtype0
M
strided_slice_66/stack_2Const*
dtype0*
valueB"      
?
strided_slice_66StridedSlicestrided_slice_49strided_slice_66/stackstrided_slice_66/stack_1strided_slice_66/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
:
GatherV2_24/axisConst*
dtype0*
value	B : 
?
GatherV2_24GatherV2strided_slice_48strided_slice_66GatherV2_24/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
2
mul_19MulExpandDims_18GatherV2_24*
T0
&
add_5AddV2add_4mul_19*
T0

GWyIdentityadd_5*
T0
K
strided_slice_67/stackConst*
valueB"       *
dtype0
M
strided_slice_67/stack_1Const*
valueB"       *
dtype0
M
strided_slice_67/stack_2Const*
dtype0*
valueB"      
?
strided_slice_67StridedSlicemul_2strided_slice_67/stackstrided_slice_67/stack_1strided_slice_67/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask
D
ExpandDims_19/dimConst*
valueB :
?????????*
dtype0
U
ExpandDims_19
ExpandDimsstrided_slice_67ExpandDims_19/dim*
T0*

Tdim0
K
strided_slice_68/stackConst*
valueB"        *
dtype0
M
strided_slice_68/stack_1Const*
valueB"       *
dtype0
M
strided_slice_68/stack_2Const*
dtype0*
valueB"      
?
strided_slice_68StridedSlicestrided_slice_49strided_slice_68/stackstrided_slice_68/stack_1strided_slice_68/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
:
GatherV2_25/axisConst*
dtype0*
value	B : 
?
GatherV2_25GatherV2strided_slice_48strided_slice_68GatherV2_25/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0
2
mul_20MulExpandDims_19GatherV2_25*
T0
K
strided_slice_69/stackConst*
valueB"       *
dtype0
M
strided_slice_69/stack_1Const*
valueB"       *
dtype0
M
strided_slice_69/stack_2Const*
valueB"      *
dtype0
?
strided_slice_69StridedSlicemul_5strided_slice_69/stackstrided_slice_69/stack_1strided_slice_69/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
D
ExpandDims_20/dimConst*
dtype0*
valueB :
?????????
U
ExpandDims_20
ExpandDimsstrided_slice_69ExpandDims_20/dim*

Tdim0*
T0
K
strided_slice_70/stackConst*
dtype0*
valueB"       
M
strided_slice_70/stack_1Const*
valueB"       *
dtype0
M
strided_slice_70/stack_2Const*
valueB"      *
dtype0
?
strided_slice_70StridedSlicestrided_slice_49strided_slice_70/stackstrided_slice_70/stack_1strided_slice_70/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
:
GatherV2_26/axisConst*
value	B : *
dtype0
?
GatherV2_26GatherV2strided_slice_48strided_slice_70GatherV2_26/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
2
mul_21MulExpandDims_20GatherV2_26*
T0
'
add_6AddV2mul_20mul_21*
T0
K
strided_slice_71/stackConst*
valueB"       *
dtype0
M
strided_slice_71/stack_1Const*
dtype0*
valueB"       
M
strided_slice_71/stack_2Const*
valueB"      *
dtype0
?
strided_slice_71StridedSlicemul_8strided_slice_71/stackstrided_slice_71/stack_1strided_slice_71/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
shrink_axis_mask
D
ExpandDims_21/dimConst*
valueB :
?????????*
dtype0
U
ExpandDims_21
ExpandDimsstrided_slice_71ExpandDims_21/dim*

Tdim0*
T0
K
strided_slice_72/stackConst*
valueB"       *
dtype0
M
strided_slice_72/stack_1Const*
valueB"       *
dtype0
M
strided_slice_72/stack_2Const*
dtype0*
valueB"      
?
strided_slice_72StridedSlicestrided_slice_49strided_slice_72/stackstrided_slice_72/stack_1strided_slice_72/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
:
GatherV2_27/axisConst*
dtype0*
value	B : 
?
GatherV2_27GatherV2strided_slice_48strided_slice_72GatherV2_27/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
2
mul_22MulExpandDims_21GatherV2_27*
T0
&
add_7AddV2add_6mul_22*
T0
K
strided_slice_73/stackConst*
valueB"       *
dtype0
M
strided_slice_73/stack_1Const*
dtype0*
valueB"       
M
strided_slice_73/stack_2Const*
dtype0*
valueB"      
?
strided_slice_73StridedSlicemul_11strided_slice_73/stackstrided_slice_73/stack_1strided_slice_73/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
D
ExpandDims_22/dimConst*
valueB :
?????????*
dtype0
U
ExpandDims_22
ExpandDimsstrided_slice_73ExpandDims_22/dim*
T0*

Tdim0
K
strided_slice_74/stackConst*
valueB"       *
dtype0
M
strided_slice_74/stack_1Const*
valueB"       *
dtype0
M
strided_slice_74/stack_2Const*
valueB"      *
dtype0
?
strided_slice_74StridedSlicestrided_slice_49strided_slice_74/stackstrided_slice_74/stack_1strided_slice_74/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
:
GatherV2_28/axisConst*
value	B : *
dtype0
?
GatherV2_28GatherV2strided_slice_48strided_slice_74GatherV2_28/axis*

batch_dims *
Tindices0*
Tparams0*
Taxis0
2
mul_23MulExpandDims_22GatherV2_28*
T0
&
add_8AddV2add_7mul_23*
T0

GWzIdentityadd_8*
T0
5
concat/axisConst*
dtype0*
value	B : 
L
concatConcatV2GWxGWyGWzconcat/axis*
N*

Tidx0*
T0

GWIdentityconcat*
T0"?