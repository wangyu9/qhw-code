
;
VPlaceholder*
dtype0*
shape:?????????
;
FPlaceholder*
dtype0	*
shape:?????????
D
WPlaceholder*
dtype0*%
shape:??????????????????
8
ExpandDims/dimConst*
value	B : *
dtype0
@

ExpandDims
ExpandDimsVExpandDims/dim*
T0*

Tdim0
:
ExpandDims_1/dimConst*
dtype0*
value	B : 
D
ExpandDims_1
ExpandDimsFExpandDims_1/dim*
T0	*

Tdim0
:
ExpandDims_2/dimConst*
value	B : *
dtype0
D
ExpandDims_2
ExpandDimsWExpandDims_2/dim*

Tdim0*
T0
J
Sum/reduction_indicesConst*
dtype0*
valueB"       
S
SumSum
ExpandDimsSum/reduction_indices*
T0*

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
ExpandDimsstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
N
strided_slice_1/stackConst*!
valueB"            *
dtype0
P
strided_slice_1/stack_1Const*!
valueB"          *
dtype0
P
strided_slice_1/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_1StridedSliceExpandDims_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0	*
Index0
7
GatherV2/axisConst*
value	B : *
dtype0
?
GatherV2GatherV2strided_slicestrided_slice_1GatherV2/axis*
Taxis0*

batch_dims *
Tindices0	*
Tparams0
N
strided_slice_2/stackConst*!
valueB"            *
dtype0
P
strided_slice_2/stack_1Const*
dtype0*!
valueB"           
P
strided_slice_2/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_2StridedSlice
ExpandDimsstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
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
strided_slice_3StridedSliceExpandDims_1strided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
end_mask*
Index0*
T0	*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask 
9
GatherV2_1/axisConst*
dtype0*
value	B : 
?

GatherV2_1GatherV2strided_slice_2strided_slice_3GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0	*
Tparams0
N
strided_slice_4/stackConst*!
valueB"            *
dtype0
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
ExpandDimsstrided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask
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
strided_slice_5StridedSliceExpandDims_1strided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*
T0	*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
9
GatherV2_2/axisConst*
value	B : *
dtype0
?

GatherV2_2GatherV2strided_slice_4strided_slice_5GatherV2_2/axis*
Taxis0*

batch_dims *
Tindices0	*
Tparams0
M
stackPackGatherV2
GatherV2_1
GatherV2_2*
T0*

axis*
N
4
stack_1Packstack*
T0*

axis *
N
R
strided_slice_6/stackConst*%
valueB"                *
dtype0
T
strided_slice_6/stack_1Const*%
valueB"              *
dtype0
T
strided_slice_6/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_6StridedSlicestack_1strided_slice_6/stackstrided_slice_6/stack_1strided_slice_6/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask

R
strided_slice_7/stackConst*%
valueB"               *
dtype0
T
strided_slice_7/stack_1Const*
dtype0*%
valueB"              
T
strided_slice_7/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_7StridedSlicestack_1strided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
*
new_axis_mask *
end_mask

R
strided_slice_8/stackConst*%
valueB"               *
dtype0
T
strided_slice_8/stack_1Const*%
valueB"              *
dtype0
T
strided_slice_8/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_8StridedSlicestack_1strided_slice_8/stackstrided_slice_8/stack_1strided_slice_8/stack_2*
end_mask
*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
*
new_axis_mask 
5
subSubstrided_slice_8strided_slice_6*
T0
7
sub_1Substrided_slice_7strided_slice_6*
T0

mulMulsubsub_1*
T0
J
Sum_1/reduction_indicesConst*
valueB :
?????????*
dtype0
P
Sum_1SummulSum_1/reduction_indices*

Tidx0*
	keep_dims(*
T0
J
strided_slice_9/stackConst*
valueB"        *
dtype0
L
strided_slice_9/stack_1Const*
valueB"       *
dtype0
L
strided_slice_9/stack_2Const*
dtype0*
valueB"      
?
strided_slice_9StridedSlicesubstrided_slice_9/stackstrided_slice_9/stack_1strided_slice_9/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
K
strided_slice_10/stackConst*
dtype0*
valueB"       
M
strided_slice_10/stack_1Const*
valueB"       *
dtype0
M
strided_slice_10/stack_2Const*
valueB"      *
dtype0
?
strided_slice_10StridedSlicesub_1strided_slice_10/stackstrided_slice_10/stack_1strided_slice_10/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
8
mul_1Mulstrided_slice_9strided_slice_10*
T0
K
strided_slice_11/stackConst*
valueB"       *
dtype0
M
strided_slice_11/stack_1Const*
valueB"       *
dtype0
M
strided_slice_11/stack_2Const*
valueB"      *
dtype0
?
strided_slice_11StridedSlicesubstrided_slice_11/stackstrided_slice_11/stack_1strided_slice_11/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
shrink_axis_mask
K
strided_slice_12/stackConst*
valueB"        *
dtype0
M
strided_slice_12/stack_1Const*
valueB"       *
dtype0
M
strided_slice_12/stack_2Const*
valueB"      *
dtype0
?
strided_slice_12StridedSlicesub_1strided_slice_12/stackstrided_slice_12/stack_1strided_slice_12/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
9
mul_2Mulstrided_slice_11strided_slice_12*
T0
#
sub_2Submul_1mul_2*
T0
 
SquareSquaresub_2*
T0

mul_3Mulsubsub*
T0
J
Sum_2/reduction_indicesConst*
dtype0*
valueB :
?????????
R
Sum_2Summul_3Sum_2/reduction_indices*

Tidx0*
	keep_dims(*
T0
&
divRealDivSum_2Square*
T0
!
mul_4Mulsub_1div*
T0
#
mul_5Mulsub_1sub_1*
T0
J
Sum_3/reduction_indicesConst*
valueB :
?????????*
dtype0
R
Sum_3Summul_5Sum_3/reduction_indices*
T0*

Tidx0*
	keep_dims(
(
div_1RealDivSum_3Square*
T0
!
mul_6Mulsubdiv_1*
T0
#
addAddV2mul_4mul_6*
T0
#
add_1AddV2sub_1sub*
T0
(
div_2RealDivSum_1Square*
T0
#
mul_7Muladd_1div_2*
T0
!
sub_3Subaddmul_7*
T0
O
strided_slice_13/stackConst*!
valueB"            *
dtype0
Q
strided_slice_13/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_13/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_13StridedSlice
ExpandDimsstrided_slice_13/stackstrided_slice_13/stack_1strided_slice_13/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0
O
strided_slice_14/stackConst*!
valueB"            *
dtype0
Q
strided_slice_14/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_14/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_14StridedSliceExpandDims_1strided_slice_14/stackstrided_slice_14/stack_1strided_slice_14/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0	
9
GatherV2_3/axisConst*
value	B : *
dtype0
?

GatherV2_3GatherV2strided_slice_13strided_slice_14GatherV2_3/axis*

batch_dims *
Tindices0	*
Tparams0*
Taxis0
O
strided_slice_15/stackConst*!
valueB"            *
dtype0
Q
strided_slice_15/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_15/stack_2Const*
dtype0*!
valueB"         
?
strided_slice_15StridedSlice
ExpandDimsstrided_slice_15/stackstrided_slice_15/stack_1strided_slice_15/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0
O
strided_slice_16/stackConst*!
valueB"           *
dtype0
Q
strided_slice_16/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_16/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_16StridedSliceExpandDims_1strided_slice_16/stackstrided_slice_16/stack_1strided_slice_16/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0	*
Index0
9
GatherV2_4/axisConst*
value	B : *
dtype0
?

GatherV2_4GatherV2strided_slice_15strided_slice_16GatherV2_4/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0	
O
strided_slice_17/stackConst*!
valueB"            *
dtype0
Q
strided_slice_17/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_17/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_17StridedSlice
ExpandDimsstrided_slice_17/stackstrided_slice_17/stack_1strided_slice_17/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0
O
strided_slice_18/stackConst*!
valueB"           *
dtype0
Q
strided_slice_18/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_18/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_18StridedSliceExpandDims_1strided_slice_18/stackstrided_slice_18/stack_1strided_slice_18/stack_2*
Index0*
T0	*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
9
GatherV2_5/axisConst*
value	B : *
dtype0
?

GatherV2_5GatherV2strided_slice_17strided_slice_18GatherV2_5/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0	
Q
stack_2Pack
GatherV2_3
GatherV2_4
GatherV2_5*
N*
T0*

axis
6
stack_3Packstack_2*
T0*

axis *
N
S
strided_slice_19/stackConst*
dtype0*%
valueB"               
U
strided_slice_19/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_19/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_19StridedSlicestack_3strided_slice_19/stackstrided_slice_19/stack_1strided_slice_19/stack_2*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
T0*
Index0
S
strided_slice_20/stackConst*%
valueB"               *
dtype0
U
strided_slice_20/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_20/stack_2Const*
dtype0*%
valueB"            
?
strided_slice_20StridedSlicestack_3strided_slice_20/stackstrided_slice_20/stack_1strided_slice_20/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
*
new_axis_mask *
end_mask

S
strided_slice_21/stackConst*%
valueB"                *
dtype0
U
strided_slice_21/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_21/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_21StridedSlicestack_3strided_slice_21/stackstrided_slice_21/stack_1strided_slice_21/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
*
new_axis_mask *
end_mask

9
sub_4Substrided_slice_21strided_slice_19*
T0
9
sub_5Substrided_slice_20strided_slice_19*
T0
#
mul_8Mulsub_4sub_5*
T0
J
Sum_4/reduction_indicesConst*
valueB :
?????????*
dtype0
R
Sum_4Summul_8Sum_4/reduction_indices*
T0*

Tidx0*
	keep_dims(
K
strided_slice_22/stackConst*
valueB"        *
dtype0
M
strided_slice_22/stack_1Const*
valueB"       *
dtype0
M
strided_slice_22/stack_2Const*
valueB"      *
dtype0
?
strided_slice_22StridedSlicesub_4strided_slice_22/stackstrided_slice_22/stack_1strided_slice_22/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
K
strided_slice_23/stackConst*
dtype0*
valueB"       
M
strided_slice_23/stack_1Const*
valueB"       *
dtype0
M
strided_slice_23/stack_2Const*
dtype0*
valueB"      
?
strided_slice_23StridedSlicesub_5strided_slice_23/stackstrided_slice_23/stack_1strided_slice_23/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
9
mul_9Mulstrided_slice_22strided_slice_23*
T0
K
strided_slice_24/stackConst*
dtype0*
valueB"       
M
strided_slice_24/stack_1Const*
valueB"       *
dtype0
M
strided_slice_24/stack_2Const*
valueB"      *
dtype0
?
strided_slice_24StridedSlicesub_4strided_slice_24/stackstrided_slice_24/stack_1strided_slice_24/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask 
K
strided_slice_25/stackConst*
valueB"        *
dtype0
M
strided_slice_25/stack_1Const*
valueB"       *
dtype0
M
strided_slice_25/stack_2Const*
dtype0*
valueB"      
?
strided_slice_25StridedSlicesub_5strided_slice_25/stackstrided_slice_25/stack_1strided_slice_25/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
:
mul_10Mulstrided_slice_24strided_slice_25*
T0
$
sub_6Submul_9mul_10*
T0
"
Square_1Squaresub_6*
T0
$
mul_11Mulsub_4sub_4*
T0
J
Sum_5/reduction_indicesConst*
valueB :
?????????*
dtype0
S
Sum_5Summul_11Sum_5/reduction_indices*
T0*

Tidx0*
	keep_dims(
*
div_3RealDivSum_5Square_1*
T0
$
mul_12Mulsub_5div_3*
T0
$
mul_13Mulsub_5sub_5*
T0
J
Sum_6/reduction_indicesConst*
dtype0*
valueB :
?????????
S
Sum_6Summul_13Sum_6/reduction_indices*

Tidx0*
	keep_dims(*
T0
*
div_4RealDivSum_6Square_1*
T0
$
mul_14Mulsub_4div_4*
T0
'
add_2AddV2mul_12mul_14*
T0
%
add_3AddV2sub_5sub_4*
T0
*
div_5RealDivSum_4Square_1*
T0
$
mul_15Muladd_3div_5*
T0
$
sub_7Subadd_2mul_15*
T0
O
strided_slice_26/stackConst*!
valueB"            *
dtype0
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
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
O
strided_slice_27/stackConst*!
valueB"            *
dtype0
Q
strided_slice_27/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_27/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_27StridedSliceExpandDims_1strided_slice_27/stackstrided_slice_27/stack_1strided_slice_27/stack_2*
end_mask*
Index0*
T0	*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask 
9
GatherV2_6/axisConst*
value	B : *
dtype0
?

GatherV2_6GatherV2strided_slice_26strided_slice_27GatherV2_6/axis*
Taxis0*

batch_dims *
Tindices0	*
Tparams0
O
strided_slice_28/stackConst*!
valueB"            *
dtype0
Q
strided_slice_28/stack_1Const*!
valueB"           *
dtype0
Q
strided_slice_28/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_28StridedSlice
ExpandDimsstrided_slice_28/stackstrided_slice_28/stack_1strided_slice_28/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
O
strided_slice_29/stackConst*!
valueB"           *
dtype0
Q
strided_slice_29/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_29/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_29StridedSliceExpandDims_1strided_slice_29/stackstrided_slice_29/stack_1strided_slice_29/stack_2*
Index0*
T0	*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
9
GatherV2_7/axisConst*
dtype0*
value	B : 
?

GatherV2_7GatherV2strided_slice_28strided_slice_29GatherV2_7/axis*
Taxis0*

batch_dims *
Tindices0	*
Tparams0
O
strided_slice_30/stackConst*
dtype0*!
valueB"            
Q
strided_slice_30/stack_1Const*
dtype0*!
valueB"           
Q
strided_slice_30/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_30StridedSlice
ExpandDimsstrided_slice_30/stackstrided_slice_30/stack_1strided_slice_30/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
O
strided_slice_31/stackConst*!
valueB"           *
dtype0
Q
strided_slice_31/stack_1Const*!
valueB"          *
dtype0
Q
strided_slice_31/stack_2Const*!
valueB"         *
dtype0
?
strided_slice_31StridedSliceExpandDims_1strided_slice_31/stackstrided_slice_31/stack_1strided_slice_31/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0	
9
GatherV2_8/axisConst*
value	B : *
dtype0
?

GatherV2_8GatherV2strided_slice_30strided_slice_31GatherV2_8/axis*

batch_dims *
Tindices0	*
Tparams0*
Taxis0
Q
stack_4Pack
GatherV2_6
GatherV2_7
GatherV2_8*
T0*

axis*
N
6
stack_5Packstack_4*
T0*

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
strided_slice_32/stack_2Const*%
valueB"            *
dtype0
?
strided_slice_32StridedSlicestack_5strided_slice_32/stackstrided_slice_32/stack_1strided_slice_32/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
*
new_axis_mask *
end_mask

S
strided_slice_33/stackConst*%
valueB"                *
dtype0
U
strided_slice_33/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_33/stack_2Const*
dtype0*%
valueB"            
?
strided_slice_33StridedSlicestack_5strided_slice_33/stackstrided_slice_33/stack_1strided_slice_33/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask
*
new_axis_mask *
end_mask

S
strided_slice_34/stackConst*%
valueB"               *
dtype0
U
strided_slice_34/stack_1Const*%
valueB"              *
dtype0
U
strided_slice_34/stack_2Const*
dtype0*%
valueB"            
?
strided_slice_34StridedSlicestack_5strided_slice_34/stackstrided_slice_34/stack_1strided_slice_34/stack_2*
shrink_axis_mask*

begin_mask
*
ellipsis_mask *
new_axis_mask *
end_mask
*
T0*
Index0
9
sub_8Substrided_slice_34strided_slice_32*
T0
9
sub_9Substrided_slice_33strided_slice_32*
T0
$
mul_16Mulsub_8sub_9*
T0
J
Sum_7/reduction_indicesConst*
valueB :
?????????*
dtype0
S
Sum_7Summul_16Sum_7/reduction_indices*
T0*

Tidx0*
	keep_dims(
K
strided_slice_35/stackConst*
valueB"        *
dtype0
M
strided_slice_35/stack_1Const*
dtype0*
valueB"       
M
strided_slice_35/stack_2Const*
valueB"      *
dtype0
?
strided_slice_35StridedSlicesub_8strided_slice_35/stackstrided_slice_35/stack_1strided_slice_35/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
K
strided_slice_36/stackConst*
valueB"       *
dtype0
M
strided_slice_36/stack_1Const*
dtype0*
valueB"       
M
strided_slice_36/stack_2Const*
valueB"      *
dtype0
?
strided_slice_36StridedSlicesub_9strided_slice_36/stackstrided_slice_36/stack_1strided_slice_36/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
:
mul_17Mulstrided_slice_35strided_slice_36*
T0
K
strided_slice_37/stackConst*
valueB"       *
dtype0
M
strided_slice_37/stack_1Const*
dtype0*
valueB"       
M
strided_slice_37/stack_2Const*
valueB"      *
dtype0
?
strided_slice_37StridedSlicesub_8strided_slice_37/stackstrided_slice_37/stack_1strided_slice_37/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
K
strided_slice_38/stackConst*
valueB"        *
dtype0
M
strided_slice_38/stack_1Const*
dtype0*
valueB"       
M
strided_slice_38/stack_2Const*
valueB"      *
dtype0
?
strided_slice_38StridedSlicesub_9strided_slice_38/stackstrided_slice_38/stack_1strided_slice_38/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0
:
mul_18Mulstrided_slice_37strided_slice_38*
T0
&
sub_10Submul_17mul_18*
T0
#
Square_2Squaresub_10*
T0
$
mul_19Mulsub_8sub_8*
T0
J
Sum_8/reduction_indicesConst*
valueB :
?????????*
dtype0
S
Sum_8Summul_19Sum_8/reduction_indices*

Tidx0*
	keep_dims(*
T0
*
div_6RealDivSum_8Square_2*
T0
$
mul_20Mulsub_9div_6*
T0
$
mul_21Mulsub_9sub_9*
T0
J
Sum_9/reduction_indicesConst*
valueB :
?????????*
dtype0
S
Sum_9Summul_21Sum_9/reduction_indices*
T0*

Tidx0*
	keep_dims(
*
div_7RealDivSum_9Square_2*
T0
$
mul_22Mulsub_8div_7*
T0
'
add_4AddV2mul_20mul_22*
T0
%
add_5AddV2sub_9sub_8*
T0
*
div_8RealDivSum_7Square_2*
T0
$
mul_23Muladd_5div_8*
T0
%
sub_11Subadd_4mul_23*
T0
D
strided_slice_39/stackConst*
valueB: *
dtype0
F
strided_slice_39/stack_1Const*
dtype0*
valueB:
F
strided_slice_39/stack_2Const*
dtype0*
valueB:
?
strided_slice_39StridedSliceExpandDims_2strided_slice_39/stackstrided_slice_39/stack_1strided_slice_39/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
D
strided_slice_40/stackConst*
valueB: *
dtype0
F
strided_slice_40/stack_1Const*
valueB:*
dtype0
F
strided_slice_40/stack_2Const*
valueB:*
dtype0
?
strided_slice_40StridedSliceExpandDims_1strided_slice_40/stackstrided_slice_40/stack_1strided_slice_40/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0	
K
strided_slice_41/stackConst*
valueB"        *
dtype0
M
strided_slice_41/stack_1Const*
dtype0*
valueB"       
M
strided_slice_41/stack_2Const*
valueB"      *
dtype0
?
strided_slice_41StridedSlicestrided_slice_40strided_slice_41/stackstrided_slice_41/stack_1strided_slice_41/stack_2*
T0	*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
9
GatherV2_9/axisConst*
value	B : *
dtype0
?

GatherV2_9GatherV2strided_slice_39strided_slice_41GatherV2_9/axis*

batch_dims *
Tindices0	*
Tparams0*
Taxis0
K
strided_slice_42/stackConst*
valueB"        *
dtype0
M
strided_slice_42/stack_1Const*
valueB"       *
dtype0
M
strided_slice_42/stack_2Const*
valueB"      *
dtype0
?
strided_slice_42StridedSlicesub_3strided_slice_42/stackstrided_slice_42/stack_1strided_slice_42/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
C
ExpandDims_3/dimConst*
valueB :
?????????*
dtype0
S
ExpandDims_3
ExpandDimsstrided_slice_42ExpandDims_3/dim*

Tdim0*
T0
K
strided_slice_43/stackConst*
dtype0*
valueB"        
M
strided_slice_43/stack_1Const*
valueB"       *
dtype0
M
strided_slice_43/stack_2Const*
valueB"      *
dtype0
?
strided_slice_43StridedSlicestrided_slice_40strided_slice_43/stackstrided_slice_43/stack_1strided_slice_43/stack_2*
T0	*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
:
GatherV2_10/axisConst*
value	B : *
dtype0
?
GatherV2_10GatherV2strided_slice_39strided_slice_43GatherV2_10/axis*

batch_dims *
Tindices0	*
Tparams0*
Taxis0
1
mul_24MulExpandDims_3GatherV2_10*
T0
K
strided_slice_44/stackConst*
valueB"        *
dtype0
M
strided_slice_44/stack_1Const*
valueB"       *
dtype0
M
strided_slice_44/stack_2Const*
dtype0*
valueB"      
?
strided_slice_44StridedSlicesub_7strided_slice_44/stackstrided_slice_44/stack_1strided_slice_44/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
C
ExpandDims_4/dimConst*
valueB :
?????????*
dtype0
S
ExpandDims_4
ExpandDimsstrided_slice_44ExpandDims_4/dim*

Tdim0*
T0
K
strided_slice_45/stackConst*
dtype0*
valueB"       
M
strided_slice_45/stack_1Const*
valueB"       *
dtype0
M
strided_slice_45/stack_2Const*
valueB"      *
dtype0
?
strided_slice_45StridedSlicestrided_slice_40strided_slice_45/stackstrided_slice_45/stack_1strided_slice_45/stack_2*
T0	*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
:
GatherV2_11/axisConst*
value	B : *
dtype0
?
GatherV2_11GatherV2strided_slice_39strided_slice_45GatherV2_11/axis*
Taxis0*

batch_dims *
Tindices0	*
Tparams0
1
mul_25MulExpandDims_4GatherV2_11*
T0
'
add_6AddV2mul_24mul_25*
T0
K
strided_slice_46/stackConst*
valueB"        *
dtype0
M
strided_slice_46/stack_1Const*
valueB"       *
dtype0
M
strided_slice_46/stack_2Const*
valueB"      *
dtype0
?
strided_slice_46StridedSlicesub_11strided_slice_46/stackstrided_slice_46/stack_1strided_slice_46/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0
C
ExpandDims_5/dimConst*
valueB :
?????????*
dtype0
S
ExpandDims_5
ExpandDimsstrided_slice_46ExpandDims_5/dim*
T0*

Tdim0
K
strided_slice_47/stackConst*
valueB"       *
dtype0
M
strided_slice_47/stack_1Const*
valueB"       *
dtype0
M
strided_slice_47/stack_2Const*
valueB"      *
dtype0
?
strided_slice_47StridedSlicestrided_slice_40strided_slice_47/stackstrided_slice_47/stack_1strided_slice_47/stack_2*
end_mask*
Index0*
T0	*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask 
:
GatherV2_12/axisConst*
value	B : *
dtype0
?
GatherV2_12GatherV2strided_slice_39strided_slice_47GatherV2_12/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0	
1
mul_26MulExpandDims_5GatherV2_12*
T0
&
add_7AddV2add_6mul_26*
T0

GWxIdentityadd_7*
T0
K
strided_slice_48/stackConst*
valueB"       *
dtype0
M
strided_slice_48/stack_1Const*
valueB"       *
dtype0
M
strided_slice_48/stack_2Const*
valueB"      *
dtype0
?
strided_slice_48StridedSlicesub_3strided_slice_48/stackstrided_slice_48/stack_1strided_slice_48/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0
C
ExpandDims_6/dimConst*
dtype0*
valueB :
?????????
S
ExpandDims_6
ExpandDimsstrided_slice_48ExpandDims_6/dim*

Tdim0*
T0
K
strided_slice_49/stackConst*
dtype0*
valueB"        
M
strided_slice_49/stack_1Const*
valueB"       *
dtype0
M
strided_slice_49/stack_2Const*
dtype0*
valueB"      
?
strided_slice_49StridedSlicestrided_slice_40strided_slice_49/stackstrided_slice_49/stack_1strided_slice_49/stack_2*
T0	*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
:
GatherV2_13/axisConst*
value	B : *
dtype0
?
GatherV2_13GatherV2strided_slice_39strided_slice_49GatherV2_13/axis*

batch_dims *
Tindices0	*
Tparams0*
Taxis0
1
mul_27MulExpandDims_6GatherV2_13*
T0
K
strided_slice_50/stackConst*
valueB"       *
dtype0
M
strided_slice_50/stack_1Const*
valueB"       *
dtype0
M
strided_slice_50/stack_2Const*
dtype0*
valueB"      
?
strided_slice_50StridedSlicesub_7strided_slice_50/stackstrided_slice_50/stack_1strided_slice_50/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
C
ExpandDims_7/dimConst*
valueB :
?????????*
dtype0
S
ExpandDims_7
ExpandDimsstrided_slice_50ExpandDims_7/dim*

Tdim0*
T0
K
strided_slice_51/stackConst*
valueB"       *
dtype0
M
strided_slice_51/stack_1Const*
valueB"       *
dtype0
M
strided_slice_51/stack_2Const*
valueB"      *
dtype0
?
strided_slice_51StridedSlicestrided_slice_40strided_slice_51/stackstrided_slice_51/stack_1strided_slice_51/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0	*
Index0
:
GatherV2_14/axisConst*
value	B : *
dtype0
?
GatherV2_14GatherV2strided_slice_39strided_slice_51GatherV2_14/axis*

batch_dims *
Tindices0	*
Tparams0*
Taxis0
1
mul_28MulExpandDims_7GatherV2_14*
T0
'
add_8AddV2mul_27mul_28*
T0
K
strided_slice_52/stackConst*
valueB"       *
dtype0
M
strided_slice_52/stack_1Const*
dtype0*
valueB"       
M
strided_slice_52/stack_2Const*
dtype0*
valueB"      
?
strided_slice_52StridedSlicesub_11strided_slice_52/stackstrided_slice_52/stack_1strided_slice_52/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
C
ExpandDims_8/dimConst*
dtype0*
valueB :
?????????
S
ExpandDims_8
ExpandDimsstrided_slice_52ExpandDims_8/dim*

Tdim0*
T0
K
strided_slice_53/stackConst*
dtype0*
valueB"       
M
strided_slice_53/stack_1Const*
valueB"       *
dtype0
M
strided_slice_53/stack_2Const*
dtype0*
valueB"      
?
strided_slice_53StridedSlicestrided_slice_40strided_slice_53/stackstrided_slice_53/stack_1strided_slice_53/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0	
:
GatherV2_15/axisConst*
value	B : *
dtype0
?
GatherV2_15GatherV2strided_slice_39strided_slice_53GatherV2_15/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0	
1
mul_29MulExpandDims_8GatherV2_15*
T0
&
add_9AddV2add_8mul_29*
T0

GWyIdentityadd_9*
T0
5
concat/axisConst*
value	B : *
dtype0
G
concatConcatV2GWxGWyconcat/axis*

Tidx0*
T0*
N

GWIdentityconcat*
T0"?