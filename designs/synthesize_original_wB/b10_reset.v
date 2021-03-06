module b10_reset(clock, RESET_G, nRESET_G, R_BUTTON, G_BUTTON, KEY, START, TEST, RTS, RTR, V_IN_3_, V_IN_2_, V_IN_1_, V_IN_0_, CTS_REG, CTR_REG, V_OUT_REG_3_, V_OUT_REG_2_, V_OUT_REG_1_, V_OUT_REG_0_);

output CTR_REG;
output CTS_REG;
input G_BUTTON;
input KEY;
wire LAST_G_REG; 
wire LAST_R_REG; 
input RESET_G;
input RTR;
input RTS;
input R_BUTTON;
wire SIGN_REG_3_; 
input START;
wire STATO_REG_0_; 
wire STATO_REG_1_; 
wire STATO_REG_2_; 
wire STATO_REG_3_; 
input TEST;
wire VOTO0_REG; 
wire VOTO1_REG; 
wire VOTO2_REG; 
wire VOTO3_REG; 
input V_IN_0_;
input V_IN_1_;
input V_IN_2_;
input V_IN_3_;
output V_OUT_REG_0_;
output V_OUT_REG_1_;
output V_OUT_REG_2_;
output V_OUT_REG_3_;
wire _abc_1116_new_n100_; 
wire _abc_1116_new_n101_; 
wire _abc_1116_new_n102_; 
wire _abc_1116_new_n103_; 
wire _abc_1116_new_n104_; 
wire _abc_1116_new_n105_; 
wire _abc_1116_new_n106_; 
wire _abc_1116_new_n107_; 
wire _abc_1116_new_n108_; 
wire _abc_1116_new_n109_; 
wire _abc_1116_new_n110_; 
wire _abc_1116_new_n111_; 
wire _abc_1116_new_n112_; 
wire _abc_1116_new_n113_; 
wire _abc_1116_new_n114_; 
wire _abc_1116_new_n115_; 
wire _abc_1116_new_n116_; 
wire _abc_1116_new_n117_; 
wire _abc_1116_new_n119_; 
wire _abc_1116_new_n120_; 
wire _abc_1116_new_n121_; 
wire _abc_1116_new_n122_; 
wire _abc_1116_new_n123_; 
wire _abc_1116_new_n125_; 
wire _abc_1116_new_n126_; 
wire _abc_1116_new_n127_; 
wire _abc_1116_new_n128_; 
wire _abc_1116_new_n130_; 
wire _abc_1116_new_n132_; 
wire _abc_1116_new_n133_; 
wire _abc_1116_new_n135_; 
wire _abc_1116_new_n136_; 
wire _abc_1116_new_n138_; 
wire _abc_1116_new_n140_; 
wire _abc_1116_new_n141_; 
wire _abc_1116_new_n142_; 
wire _abc_1116_new_n143_; 
wire _abc_1116_new_n144_; 
wire _abc_1116_new_n145_; 
wire _abc_1116_new_n146_; 
wire _abc_1116_new_n147_; 
wire _abc_1116_new_n148_; 
wire _abc_1116_new_n149_; 
wire _abc_1116_new_n150_; 
wire _abc_1116_new_n151_; 
wire _abc_1116_new_n152_; 
wire _abc_1116_new_n153_; 
wire _abc_1116_new_n154_; 
wire _abc_1116_new_n155_; 
wire _abc_1116_new_n156_; 
wire _abc_1116_new_n157_; 
wire _abc_1116_new_n158_; 
wire _abc_1116_new_n159_; 
wire _abc_1116_new_n161_; 
wire _abc_1116_new_n162_; 
wire _abc_1116_new_n163_; 
wire _abc_1116_new_n164_; 
wire _abc_1116_new_n165_; 
wire _abc_1116_new_n166_; 
wire _abc_1116_new_n167_; 
wire _abc_1116_new_n168_; 
wire _abc_1116_new_n169_; 
wire _abc_1116_new_n170_; 
wire _abc_1116_new_n171_; 
wire _abc_1116_new_n172_; 
wire _abc_1116_new_n173_; 
wire _abc_1116_new_n175_; 
wire _abc_1116_new_n176_; 
wire _abc_1116_new_n177_; 
wire _abc_1116_new_n179_; 
wire _abc_1116_new_n180_; 
wire _abc_1116_new_n181_; 
wire _abc_1116_new_n182_; 
wire _abc_1116_new_n183_; 
wire _abc_1116_new_n184_; 
wire _abc_1116_new_n185_; 
wire _abc_1116_new_n187_; 
wire _abc_1116_new_n189_; 
wire _abc_1116_new_n190_; 
wire _abc_1116_new_n191_; 
wire _abc_1116_new_n192_; 
wire _abc_1116_new_n193_; 
wire _abc_1116_new_n194_; 
wire _abc_1116_new_n195_; 
wire _abc_1116_new_n196_; 
wire _abc_1116_new_n198_; 
wire _abc_1116_new_n199_; 
wire _abc_1116_new_n200_; 
wire _abc_1116_new_n201_; 
wire _abc_1116_new_n202_; 
wire _abc_1116_new_n203_; 
wire _abc_1116_new_n204_; 
wire _abc_1116_new_n205_; 
wire _abc_1116_new_n206_; 
wire _abc_1116_new_n208_; 
wire _abc_1116_new_n47_; 
wire _abc_1116_new_n48_; 
wire _abc_1116_new_n49_; 
wire _abc_1116_new_n50_; 
wire _abc_1116_new_n51_; 
wire _abc_1116_new_n53_; 
wire _abc_1116_new_n54_; 
wire _abc_1116_new_n55_; 
wire _abc_1116_new_n56_; 
wire _abc_1116_new_n57_; 
wire _abc_1116_new_n58_; 
wire _abc_1116_new_n59_; 
wire _abc_1116_new_n60_; 
wire _abc_1116_new_n61_; 
wire _abc_1116_new_n62_; 
wire _abc_1116_new_n63_; 
wire _abc_1116_new_n64_; 
wire _abc_1116_new_n65_; 
wire _abc_1116_new_n66_; 
wire _abc_1116_new_n67_; 
wire _abc_1116_new_n68_; 
wire _abc_1116_new_n69_; 
wire _abc_1116_new_n70_; 
wire _abc_1116_new_n71_; 
wire _abc_1116_new_n72_; 
wire _abc_1116_new_n73_; 
wire _abc_1116_new_n74_; 
wire _abc_1116_new_n75_; 
wire _abc_1116_new_n76_; 
wire _abc_1116_new_n77_; 
wire _abc_1116_new_n78_; 
wire _abc_1116_new_n79_; 
wire _abc_1116_new_n80_; 
wire _abc_1116_new_n81_; 
wire _abc_1116_new_n82_; 
wire _abc_1116_new_n83_; 
wire _abc_1116_new_n84_; 
wire _abc_1116_new_n85_; 
wire _abc_1116_new_n86_; 
wire _abc_1116_new_n87_; 
wire _abc_1116_new_n88_; 
wire _abc_1116_new_n90_; 
wire _abc_1116_new_n91_; 
wire _abc_1116_new_n92_; 
wire _abc_1116_new_n93_; 
wire _abc_1116_new_n94_; 
wire _abc_1116_new_n95_; 
wire _abc_1116_new_n96_; 
wire _abc_1116_new_n97_; 
wire _abc_1116_new_n98_; 
wire _abc_1116_new_n99_; 
wire _auto_iopadmap_cc_368_execute_1280; 
wire _auto_iopadmap_cc_368_execute_1282; 
wire _auto_iopadmap_cc_368_execute_1284; 
wire _auto_iopadmap_cc_368_execute_1286; 
wire _auto_iopadmap_cc_368_execute_1288; 
wire _auto_iopadmap_cc_368_execute_1290; 
input clock;
wire clock_bF_buf0; 
wire clock_bF_buf1; 
wire clock_bF_buf2; 
wire clock_bF_buf3; 
wire n100; 
wire n105; 
wire n109; 
wire n114; 
wire n40; 
wire n45; 
wire n50; 
wire n55; 
wire n60; 
wire n65; 
wire n69; 
wire n73; 
wire n77; 
wire n81; 
wire n86; 
wire n91; 
wire n95; 
input nRESET_G;
AND2X2 AND2X2_1 ( .A(STATO_REG_1_), .B(STATO_REG_0_), .Y(_abc_1116_new_n61_));
AND2X2 AND2X2_2 ( .A(_abc_1116_new_n47_), .B(START), .Y(_abc_1116_new_n103_));
AND2X2 AND2X2_3 ( .A(_abc_1116_new_n78_), .B(_abc_1116_new_n51_), .Y(_abc_1116_new_n122_));
AND2X2 AND2X2_4 ( .A(_abc_1116_new_n142_), .B(_abc_1116_new_n157_), .Y(_abc_1116_new_n158_));
AND2X2 AND2X2_5 ( .A(_abc_1116_new_n193_), .B(STATO_REG_1_), .Y(_abc_1116_new_n194_));
AOI21X1 AOI21X1_1 ( .A(_abc_1116_new_n69_), .B(STATO_REG_3_), .C(_abc_1116_new_n74_), .Y(_abc_1116_new_n75_));
AOI21X1 AOI21X1_10 ( .A(_abc_1116_new_n176_), .B(LAST_R_REG), .C(_abc_1116_new_n109_), .Y(_abc_1116_new_n177_));
AOI21X1 AOI21X1_11 ( .A(_abc_1116_new_n176_), .B(LAST_G_REG), .C(_abc_1116_new_n109_), .Y(_abc_1116_new_n187_));
AOI21X1 AOI21X1_12 ( .A(_abc_1116_new_n122_), .B(_abc_1116_new_n100_), .C(_abc_1116_new_n103_), .Y(_abc_1116_new_n189_));
AOI21X1 AOI21X1_13 ( .A(_abc_1116_new_n141_), .B(_abc_1116_new_n51_), .C(_abc_1116_new_n61_), .Y(_abc_1116_new_n199_));
AOI21X1 AOI21X1_14 ( .A(_abc_1116_new_n92_), .B(_auto_iopadmap_cc_368_execute_1280), .C(_abc_1116_new_n109_), .Y(_abc_1116_new_n208_));
AOI21X1 AOI21X1_2 ( .A(_abc_1116_new_n87_), .B(_abc_1116_new_n116_), .C(_abc_1116_new_n109_), .Y(_abc_1116_new_n117_));
AOI21X1 AOI21X1_3 ( .A(_abc_1116_new_n75_), .B(_abc_1116_new_n64_), .C(_abc_1116_new_n120_), .Y(_abc_1116_new_n121_));
AOI21X1 AOI21X1_4 ( .A(_abc_1116_new_n122_), .B(STATO_REG_1_), .C(_abc_1116_new_n81_), .Y(_abc_1116_new_n123_));
AOI21X1 AOI21X1_5 ( .A(_abc_1116_new_n57_), .B(_auto_iopadmap_cc_368_execute_1290), .C(_abc_1116_new_n109_), .Y(_abc_1116_new_n130_));
AOI21X1 AOI21X1_6 ( .A(_abc_1116_new_n57_), .B(_auto_iopadmap_cc_368_execute_1288), .C(_abc_1116_new_n109_), .Y(_abc_1116_new_n133_));
AOI21X1 AOI21X1_7 ( .A(_abc_1116_new_n57_), .B(_auto_iopadmap_cc_368_execute_1286), .C(_abc_1116_new_n109_), .Y(_abc_1116_new_n136_));
AOI21X1 AOI21X1_8 ( .A(_abc_1116_new_n57_), .B(_auto_iopadmap_cc_368_execute_1284), .C(_abc_1116_new_n109_), .Y(_abc_1116_new_n138_));
AOI21X1 AOI21X1_9 ( .A(_abc_1116_new_n85_), .B(_abc_1116_new_n141_), .C(_abc_1116_new_n56_), .Y(_abc_1116_new_n142_));
AOI22X1 AOI22X1_1 ( .A(_abc_1116_new_n71_), .B(_abc_1116_new_n90_), .C(RTS), .D(_abc_1116_new_n91_), .Y(_abc_1116_new_n92_));
AOI22X1 AOI22X1_2 ( .A(_abc_1116_new_n94_), .B(_abc_1116_new_n47_), .C(_abc_1116_new_n79_), .D(_abc_1116_new_n93_), .Y(_abc_1116_new_n95_));
AOI22X1 AOI22X1_3 ( .A(_abc_1116_new_n86_), .B(_abc_1116_new_n104_), .C(STATO_REG_0_), .D(_abc_1116_new_n103_), .Y(_abc_1116_new_n105_));
AOI22X1 AOI22X1_4 ( .A(_abc_1116_new_n85_), .B(_abc_1116_new_n141_), .C(_abc_1116_new_n77_), .D(_abc_1116_new_n91_), .Y(_abc_1116_new_n154_));
BUFX2 BUFX2_1 ( .A(_auto_iopadmap_cc_368_execute_1280), .Y(CTR_REG));
BUFX2 BUFX2_2 ( .A(_auto_iopadmap_cc_368_execute_1282), .Y(CTS_REG));
BUFX2 BUFX2_3 ( .A(_auto_iopadmap_cc_368_execute_1284), .Y(V_OUT_REG_0_));
BUFX2 BUFX2_4 ( .A(_auto_iopadmap_cc_368_execute_1286), .Y(V_OUT_REG_1_));
BUFX2 BUFX2_5 ( .A(_auto_iopadmap_cc_368_execute_1288), .Y(V_OUT_REG_2_));
BUFX2 BUFX2_6 ( .A(_auto_iopadmap_cc_368_execute_1290), .Y(V_OUT_REG_3_));
BUFX4 BUFX4_1 ( .A(clock), .Y(clock_bF_buf3));
BUFX4 BUFX4_2 ( .A(clock), .Y(clock_bF_buf2));
BUFX4 BUFX4_3 ( .A(clock), .Y(clock_bF_buf1));
BUFX4 BUFX4_4 ( .A(clock), .Y(clock_bF_buf0));
DFFPOSX1 DFFPOSX1_1 ( .CLK(clock_bF_buf3), .D(n105), .Q(_auto_iopadmap_cc_368_execute_1282));
DFFPOSX1 DFFPOSX1_10 ( .CLK(clock_bF_buf2), .D(n55), .Q(STATO_REG_1_));
DFFPOSX1 DFFPOSX1_11 ( .CLK(clock_bF_buf1), .D(n60), .Q(STATO_REG_0_));
DFFPOSX1 DFFPOSX1_12 ( .CLK(clock_bF_buf0), .D(n81), .Q(SIGN_REG_3_));
DFFPOSX1 DFFPOSX1_13 ( .CLK(clock_bF_buf3), .D(n86), .Q(VOTO1_REG));
DFFPOSX1 DFFPOSX1_14 ( .CLK(clock_bF_buf2), .D(n95), .Q(VOTO3_REG));
DFFPOSX1 DFFPOSX1_15 ( .CLK(clock_bF_buf1), .D(n100), .Q(LAST_R_REG));
DFFPOSX1 DFFPOSX1_16 ( .CLK(clock_bF_buf0), .D(n109), .Q(VOTO2_REG));
DFFPOSX1 DFFPOSX1_17 ( .CLK(clock_bF_buf3), .D(n114), .Q(LAST_G_REG));
DFFPOSX1 DFFPOSX1_2 ( .CLK(clock_bF_buf2), .D(n65), .Q(_auto_iopadmap_cc_368_execute_1290));
DFFPOSX1 DFFPOSX1_3 ( .CLK(clock_bF_buf1), .D(n77), .Q(_auto_iopadmap_cc_368_execute_1284));
DFFPOSX1 DFFPOSX1_4 ( .CLK(clock_bF_buf0), .D(n69), .Q(_auto_iopadmap_cc_368_execute_1288));
DFFPOSX1 DFFPOSX1_5 ( .CLK(clock_bF_buf3), .D(n73), .Q(_auto_iopadmap_cc_368_execute_1286));
DFFPOSX1 DFFPOSX1_6 ( .CLK(clock_bF_buf2), .D(n91), .Q(_auto_iopadmap_cc_368_execute_1280));
DFFPOSX1 DFFPOSX1_7 ( .CLK(clock_bF_buf1), .D(n40), .Q(VOTO0_REG));
DFFPOSX1 DFFPOSX1_8 ( .CLK(clock_bF_buf0), .D(n45), .Q(STATO_REG_3_));
DFFPOSX1 DFFPOSX1_9 ( .CLK(clock_bF_buf3), .D(n50), .Q(STATO_REG_2_));
INVX1 INVX1_1 ( .A(RTS), .Y(_abc_1116_new_n53_));
INVX1 INVX1_10 ( .A(VOTO0_REG), .Y(_abc_1116_new_n110_));
INVX1 INVX1_11 ( .A(VOTO3_REG), .Y(_abc_1116_new_n112_));
INVX1 INVX1_12 ( .A(VOTO2_REG), .Y(_abc_1116_new_n132_));
INVX1 INVX1_13 ( .A(VOTO1_REG), .Y(_abc_1116_new_n135_));
INVX1 INVX1_14 ( .A(KEY), .Y(_abc_1116_new_n144_));
INVX1 INVX1_15 ( .A(G_BUTTON), .Y(_abc_1116_new_n146_));
INVX1 INVX1_16 ( .A(_abc_1116_new_n148_), .Y(_abc_1116_new_n149_));
INVX1 INVX1_17 ( .A(V_IN_1_), .Y(_abc_1116_new_n150_));
INVX1 INVX1_18 ( .A(_abc_1116_new_n151_), .Y(_abc_1116_new_n152_));
INVX1 INVX1_19 ( .A(_abc_1116_new_n61_), .Y(_abc_1116_new_n161_));
INVX1 INVX1_2 ( .A(START), .Y(_abc_1116_new_n58_));
INVX1 INVX1_20 ( .A(_abc_1116_new_n163_), .Y(_abc_1116_new_n164_));
INVX1 INVX1_21 ( .A(V_IN_3_), .Y(_abc_1116_new_n165_));
INVX1 INVX1_22 ( .A(R_BUTTON), .Y(_abc_1116_new_n175_));
INVX1 INVX1_23 ( .A(_abc_1116_new_n180_), .Y(_abc_1116_new_n181_));
INVX1 INVX1_24 ( .A(V_IN_2_), .Y(_abc_1116_new_n182_));
INVX1 INVX1_3 ( .A(STATO_REG_3_), .Y(_abc_1116_new_n77_));
INVX1 INVX1_4 ( .A(_abc_1116_new_n79_), .Y(_abc_1116_new_n80_));
INVX1 INVX1_5 ( .A(_abc_1116_new_n81_), .Y(_abc_1116_new_n82_));
INVX1 INVX1_6 ( .A(_abc_1116_new_n83_), .Y(_abc_1116_new_n84_));
INVX1 INVX1_7 ( .A(_abc_1116_new_n51_), .Y(_abc_1116_new_n85_));
INVX1 INVX1_8 ( .A(V_IN_0_), .Y(_abc_1116_new_n97_));
INVX1 INVX1_9 ( .A(_abc_1116_new_n98_), .Y(_abc_1116_new_n99_));
INVX2 INVX2_1 ( .A(STATO_REG_0_), .Y(_abc_1116_new_n54_));
INVX2 INVX2_2 ( .A(RTR), .Y(_abc_1116_new_n60_));
INVX2 INVX2_3 ( .A(STATO_REG_2_), .Y(_abc_1116_new_n65_));
INVX4 INVX4_1 ( .A(STATO_REG_1_), .Y(_abc_1116_new_n71_));
INVX4 INVX4_2 ( .A(nRESET_G), .Y(_abc_1116_new_n109_));
MUX2X1 MUX2X1_1 ( .A(_abc_1116_new_n65_), .B(_abc_1116_new_n60_), .S(STATO_REG_1_), .Y(_abc_1116_new_n66_));
NAND2X1 NAND2X1_1 ( .A(_abc_1116_new_n47_), .B(_abc_1116_new_n48_), .Y(_abc_1116_new_n49_));
NAND2X1 NAND2X1_10 ( .A(VOTO2_REG), .B(_abc_1116_new_n110_), .Y(_abc_1116_new_n111_));
NAND2X1 NAND2X1_11 ( .A(VOTO1_REG), .B(_abc_1116_new_n112_), .Y(_abc_1116_new_n113_));
NAND2X1 NAND2X1_12 ( .A(_abc_1116_new_n71_), .B(_abc_1116_new_n79_), .Y(_abc_1116_new_n115_));
NAND2X1 NAND2X1_13 ( .A(_abc_1116_new_n108_), .B(_abc_1116_new_n117_), .Y(n45));
NAND2X1 NAND2X1_14 ( .A(START), .B(_abc_1116_new_n65_), .Y(_abc_1116_new_n140_));
NAND2X1 NAND2X1_15 ( .A(STATO_REG_1_), .B(KEY), .Y(_abc_1116_new_n151_));
NAND2X1 NAND2X1_16 ( .A(_abc_1116_new_n65_), .B(_abc_1116_new_n152_), .Y(_abc_1116_new_n153_));
NAND2X1 NAND2X1_17 ( .A(_abc_1116_new_n47_), .B(_abc_1116_new_n162_), .Y(_abc_1116_new_n163_));
NAND2X1 NAND2X1_18 ( .A(VOTO0_REG), .B(_abc_1116_new_n132_), .Y(_abc_1116_new_n166_));
NAND2X1 NAND2X1_19 ( .A(_abc_1116_new_n111_), .B(_abc_1116_new_n166_), .Y(_abc_1116_new_n168_));
NAND2X1 NAND2X1_2 ( .A(STATO_REG_0_), .B(STATO_REG_3_), .Y(_abc_1116_new_n51_));
NAND2X1 NAND2X1_20 ( .A(VOTO1_REG), .B(_abc_1116_new_n168_), .Y(_abc_1116_new_n169_));
NAND2X1 NAND2X1_21 ( .A(_abc_1116_new_n152_), .B(_abc_1116_new_n145_), .Y(_abc_1116_new_n176_));
NAND2X1 NAND2X1_22 ( .A(_abc_1116_new_n142_), .B(_abc_1116_new_n189_), .Y(_abc_1116_new_n190_));
NAND2X1 NAND2X1_23 ( .A(KEY), .B(_abc_1116_new_n90_), .Y(_abc_1116_new_n192_));
NAND2X1 NAND2X1_24 ( .A(STATO_REG_3_), .B(_abc_1116_new_n79_), .Y(_abc_1116_new_n198_));
NAND2X1 NAND2X1_25 ( .A(_abc_1116_new_n47_), .B(_abc_1116_new_n203_), .Y(_abc_1116_new_n204_));
NAND2X1 NAND2X1_3 ( .A(STATO_REG_0_), .B(V_IN_0_), .Y(_abc_1116_new_n68_));
NAND2X1 NAND2X1_4 ( .A(START), .B(_abc_1116_new_n47_), .Y(_abc_1116_new_n70_));
NAND2X1 NAND2X1_5 ( .A(STATO_REG_0_), .B(_abc_1116_new_n71_), .Y(_abc_1116_new_n72_));
NAND2X1 NAND2X1_6 ( .A(STATO_REG_2_), .B(_abc_1116_new_n53_), .Y(_abc_1116_new_n73_));
NAND2X1 NAND2X1_7 ( .A(_abc_1116_new_n54_), .B(_abc_1116_new_n77_), .Y(_abc_1116_new_n78_));
NAND2X1 NAND2X1_8 ( .A(_abc_1116_new_n64_), .B(_abc_1116_new_n75_), .Y(_abc_1116_new_n87_));
NAND2X1 NAND2X1_9 ( .A(STATO_REG_3_), .B(_abc_1116_new_n107_), .Y(_abc_1116_new_n108_));
NAND3X1 NAND3X1_1 ( .A(nRESET_G), .B(_abc_1116_new_n51_), .C(_abc_1116_new_n50_), .Y(n81));
NAND3X1 NAND3X1_10 ( .A(_abc_1116_new_n95_), .B(_abc_1116_new_n62_), .C(_abc_1116_new_n92_), .Y(_abc_1116_new_n96_));
NAND3X1 NAND3X1_11 ( .A(STATO_REG_0_), .B(_abc_1116_new_n64_), .C(_abc_1116_new_n75_), .Y(_abc_1116_new_n125_));
NAND3X1 NAND3X1_12 ( .A(nRESET_G), .B(_abc_1116_new_n125_), .C(_abc_1116_new_n128_), .Y(n60));
NAND3X1 NAND3X1_13 ( .A(VOTO1_REG), .B(_abc_1116_new_n148_), .C(_abc_1116_new_n158_), .Y(_abc_1116_new_n159_));
NAND3X1 NAND3X1_14 ( .A(nRESET_G), .B(_abc_1116_new_n156_), .C(_abc_1116_new_n159_), .Y(n86));
NAND3X1 NAND3X1_15 ( .A(_abc_1116_new_n135_), .B(_abc_1116_new_n111_), .C(_abc_1116_new_n166_), .Y(_abc_1116_new_n167_));
NAND3X1 NAND3X1_16 ( .A(_abc_1116_new_n61_), .B(_abc_1116_new_n167_), .C(_abc_1116_new_n169_), .Y(_abc_1116_new_n170_));
NAND3X1 NAND3X1_17 ( .A(VOTO3_REG), .B(_abc_1116_new_n163_), .C(_abc_1116_new_n158_), .Y(_abc_1116_new_n173_));
NAND3X1 NAND3X1_18 ( .A(nRESET_G), .B(_abc_1116_new_n172_), .C(_abc_1116_new_n173_), .Y(n95));
NAND3X1 NAND3X1_19 ( .A(VOTO2_REG), .B(_abc_1116_new_n180_), .C(_abc_1116_new_n158_), .Y(_abc_1116_new_n185_));
NAND3X1 NAND3X1_2 ( .A(STATO_REG_1_), .B(STATO_REG_2_), .C(_abc_1116_new_n54_), .Y(_abc_1116_new_n55_));
NAND3X1 NAND3X1_20 ( .A(nRESET_G), .B(_abc_1116_new_n184_), .C(_abc_1116_new_n185_), .Y(n109));
NAND3X1 NAND3X1_21 ( .A(nRESET_G), .B(_abc_1116_new_n195_), .C(_abc_1116_new_n196_), .Y(n40));
NAND3X1 NAND3X1_22 ( .A(_abc_1116_new_n57_), .B(_abc_1116_new_n204_), .C(_abc_1116_new_n202_), .Y(_abc_1116_new_n205_));
NAND3X1 NAND3X1_3 ( .A(STATO_REG_2_), .B(RTR), .C(_abc_1116_new_n48_), .Y(_abc_1116_new_n57_));
NAND3X1 NAND3X1_4 ( .A(STATO_REG_1_), .B(_abc_1116_new_n58_), .C(_abc_1116_new_n47_), .Y(_abc_1116_new_n59_));
NAND3X1 NAND3X1_5 ( .A(STATO_REG_2_), .B(_abc_1116_new_n60_), .C(_abc_1116_new_n61_), .Y(_abc_1116_new_n62_));
NAND3X1 NAND3X1_6 ( .A(_abc_1116_new_n57_), .B(_abc_1116_new_n59_), .C(_abc_1116_new_n62_), .Y(_abc_1116_new_n63_));
NAND3X1 NAND3X1_7 ( .A(V_IN_1_), .B(V_IN_3_), .C(V_IN_2_), .Y(_abc_1116_new_n67_));
NAND3X1 NAND3X1_8 ( .A(STATO_REG_1_), .B(_abc_1116_new_n64_), .C(_abc_1116_new_n75_), .Y(_abc_1116_new_n76_));
NAND3X1 NAND3X1_9 ( .A(_abc_1116_new_n76_), .B(_abc_1116_new_n84_), .C(_abc_1116_new_n88_), .Y(n55));
NOR2X1 NOR2X1_1 ( .A(STATO_REG_2_), .B(STATO_REG_3_), .Y(_abc_1116_new_n47_));
NOR2X1 NOR2X1_10 ( .A(STATO_REG_1_), .B(STATO_REG_2_), .Y(_abc_1116_new_n141_));
NOR2X1 NOR2X1_11 ( .A(_abc_1116_new_n140_), .B(_abc_1116_new_n78_), .Y(_abc_1116_new_n145_));
NOR2X1 NOR2X1_12 ( .A(LAST_G_REG), .B(_abc_1116_new_n146_), .Y(_abc_1116_new_n147_));
NOR2X1 NOR2X1_13 ( .A(LAST_R_REG), .B(_abc_1116_new_n175_), .Y(_abc_1116_new_n179_));
NOR2X1 NOR2X1_14 ( .A(_abc_1116_new_n97_), .B(_abc_1116_new_n154_), .Y(_abc_1116_new_n191_));
NOR2X1 NOR2X1_15 ( .A(_abc_1116_new_n60_), .B(_abc_1116_new_n72_), .Y(_abc_1116_new_n203_));
NOR2X1 NOR2X1_16 ( .A(_abc_1116_new_n109_), .B(_abc_1116_new_n205_), .Y(_abc_1116_new_n206_));
NOR2X1 NOR2X1_2 ( .A(STATO_REG_1_), .B(STATO_REG_0_), .Y(_abc_1116_new_n48_));
NOR2X1 NOR2X1_3 ( .A(_abc_1116_new_n56_), .B(_abc_1116_new_n63_), .Y(_abc_1116_new_n64_));
NOR2X1 NOR2X1_4 ( .A(STATO_REG_0_), .B(_abc_1116_new_n65_), .Y(_abc_1116_new_n79_));
NOR2X1 NOR2X1_5 ( .A(STATO_REG_1_), .B(_abc_1116_new_n54_), .Y(_abc_1116_new_n86_));
NOR2X1 NOR2X1_6 ( .A(STATO_REG_1_), .B(_abc_1116_new_n60_), .Y(_abc_1116_new_n93_));
NOR2X1 NOR2X1_7 ( .A(START), .B(_abc_1116_new_n71_), .Y(_abc_1116_new_n94_));
NOR2X1 NOR2X1_8 ( .A(STATO_REG_2_), .B(_abc_1116_new_n71_), .Y(_abc_1116_new_n100_));
NOR2X1 NOR2X1_9 ( .A(RTS), .B(_abc_1116_new_n65_), .Y(_abc_1116_new_n104_));
NOR3X1 NOR3X1_1 ( .A(STATO_REG_0_), .B(STATO_REG_2_), .C(STATO_REG_3_), .Y(_abc_1116_new_n90_));
NOR3X1 NOR3X1_2 ( .A(STATO_REG_0_), .B(_abc_1116_new_n71_), .C(_abc_1116_new_n65_), .Y(_abc_1116_new_n91_));
OAI21X1 OAI21X1_1 ( .A(TEST), .B(_abc_1116_new_n49_), .C(SIGN_REG_3_), .Y(_abc_1116_new_n50_));
OAI21X1 OAI21X1_10 ( .A(_abc_1116_new_n106_), .B(_abc_1116_new_n96_), .C(_abc_1116_new_n54_), .Y(_abc_1116_new_n107_));
OAI21X1 OAI21X1_11 ( .A(_abc_1116_new_n111_), .B(_abc_1116_new_n113_), .C(_abc_1116_new_n54_), .Y(_abc_1116_new_n119_));
OAI21X1 OAI21X1_12 ( .A(STATO_REG_1_), .B(_abc_1116_new_n54_), .C(_abc_1116_new_n119_), .Y(_abc_1116_new_n120_));
OAI21X1 OAI21X1_13 ( .A(_abc_1116_new_n65_), .B(_abc_1116_new_n121_), .C(_abc_1116_new_n123_), .Y(n50));
OAI21X1 OAI21X1_14 ( .A(STATO_REG_3_), .B(_abc_1116_new_n114_), .C(_abc_1116_new_n48_), .Y(_abc_1116_new_n126_));
OAI21X1 OAI21X1_15 ( .A(_abc_1116_new_n71_), .B(_abc_1116_new_n80_), .C(_abc_1116_new_n126_), .Y(_abc_1116_new_n127_));
OAI21X1 OAI21X1_16 ( .A(_abc_1116_new_n90_), .B(_abc_1116_new_n127_), .C(_abc_1116_new_n87_), .Y(_abc_1116_new_n128_));
OAI21X1 OAI21X1_17 ( .A(_abc_1116_new_n112_), .B(_abc_1116_new_n57_), .C(_abc_1116_new_n130_), .Y(n65));
OAI21X1 OAI21X1_18 ( .A(_abc_1116_new_n132_), .B(_abc_1116_new_n57_), .C(_abc_1116_new_n133_), .Y(n69));
OAI21X1 OAI21X1_19 ( .A(_abc_1116_new_n135_), .B(_abc_1116_new_n57_), .C(_abc_1116_new_n136_), .Y(n73));
OAI21X1 OAI21X1_2 ( .A(_abc_1116_new_n53_), .B(_abc_1116_new_n55_), .C(_abc_1116_new_n49_), .Y(_abc_1116_new_n56_));
OAI21X1 OAI21X1_20 ( .A(_abc_1116_new_n110_), .B(_abc_1116_new_n57_), .C(_abc_1116_new_n138_), .Y(n77));
OAI21X1 OAI21X1_21 ( .A(_abc_1116_new_n140_), .B(_abc_1116_new_n72_), .C(_abc_1116_new_n142_), .Y(_abc_1116_new_n143_));
OAI21X1 OAI21X1_22 ( .A(_abc_1116_new_n144_), .B(_abc_1116_new_n147_), .C(_abc_1116_new_n145_), .Y(_abc_1116_new_n148_));
OAI21X1 OAI21X1_23 ( .A(_abc_1116_new_n149_), .B(_abc_1116_new_n143_), .C(_abc_1116_new_n155_), .Y(_abc_1116_new_n156_));
OAI21X1 OAI21X1_24 ( .A(KEY), .B(_abc_1116_new_n58_), .C(_abc_1116_new_n161_), .Y(_abc_1116_new_n162_));
OAI21X1 OAI21X1_25 ( .A(_abc_1116_new_n165_), .B(_abc_1116_new_n154_), .C(_abc_1116_new_n170_), .Y(_abc_1116_new_n171_));
OAI21X1 OAI21X1_26 ( .A(_abc_1116_new_n164_), .B(_abc_1116_new_n143_), .C(_abc_1116_new_n171_), .Y(_abc_1116_new_n172_));
OAI21X1 OAI21X1_27 ( .A(_abc_1116_new_n175_), .B(_abc_1116_new_n176_), .C(_abc_1116_new_n177_), .Y(n100));
OAI21X1 OAI21X1_28 ( .A(_abc_1116_new_n144_), .B(_abc_1116_new_n179_), .C(_abc_1116_new_n145_), .Y(_abc_1116_new_n180_));
OAI21X1 OAI21X1_29 ( .A(_abc_1116_new_n181_), .B(_abc_1116_new_n143_), .C(_abc_1116_new_n183_), .Y(_abc_1116_new_n184_));
OAI21X1 OAI21X1_3 ( .A(_abc_1116_new_n71_), .B(_abc_1116_new_n80_), .C(nRESET_G), .Y(_abc_1116_new_n81_));
OAI21X1 OAI21X1_30 ( .A(_abc_1116_new_n146_), .B(_abc_1116_new_n176_), .C(_abc_1116_new_n187_), .Y(n114));
OAI21X1 OAI21X1_31 ( .A(SIGN_REG_3_), .B(_abc_1116_new_n77_), .C(_abc_1116_new_n192_), .Y(_abc_1116_new_n193_));
OAI21X1 OAI21X1_32 ( .A(_abc_1116_new_n191_), .B(_abc_1116_new_n194_), .C(_abc_1116_new_n190_), .Y(_abc_1116_new_n195_));
OAI21X1 OAI21X1_33 ( .A(STATO_REG_0_), .B(STATO_REG_3_), .C(RTR), .Y(_abc_1116_new_n200_));
OAI21X1 OAI21X1_34 ( .A(_abc_1116_new_n71_), .B(STATO_REG_2_), .C(_abc_1116_new_n200_), .Y(_abc_1116_new_n201_));
OAI21X1 OAI21X1_35 ( .A(_abc_1116_new_n201_), .B(_abc_1116_new_n199_), .C(_auto_iopadmap_cc_368_execute_1282), .Y(_abc_1116_new_n202_));
OAI21X1 OAI21X1_36 ( .A(STATO_REG_1_), .B(_abc_1116_new_n198_), .C(_abc_1116_new_n206_), .Y(n105));
OAI21X1 OAI21X1_37 ( .A(_abc_1116_new_n72_), .B(_abc_1116_new_n73_), .C(_abc_1116_new_n208_), .Y(n91));
OAI21X1 OAI21X1_4 ( .A(_abc_1116_new_n71_), .B(_abc_1116_new_n78_), .C(_abc_1116_new_n82_), .Y(_abc_1116_new_n83_));
OAI21X1 OAI21X1_5 ( .A(_abc_1116_new_n85_), .B(_abc_1116_new_n86_), .C(_abc_1116_new_n87_), .Y(_abc_1116_new_n88_));
OAI21X1 OAI21X1_6 ( .A(_abc_1116_new_n97_), .B(_abc_1116_new_n67_), .C(STATO_REG_0_), .Y(_abc_1116_new_n98_));
OAI21X1 OAI21X1_7 ( .A(STATO_REG_1_), .B(RTR), .C(_abc_1116_new_n54_), .Y(_abc_1116_new_n101_));
OAI21X1 OAI21X1_8 ( .A(_abc_1116_new_n101_), .B(_abc_1116_new_n100_), .C(STATO_REG_3_), .Y(_abc_1116_new_n102_));
OAI21X1 OAI21X1_9 ( .A(_abc_1116_new_n99_), .B(_abc_1116_new_n102_), .C(_abc_1116_new_n105_), .Y(_abc_1116_new_n106_));
OAI22X1 OAI22X1_1 ( .A(_abc_1116_new_n67_), .B(_abc_1116_new_n68_), .C(STATO_REG_0_), .D(_abc_1116_new_n66_), .Y(_abc_1116_new_n69_));
OAI22X1 OAI22X1_2 ( .A(_abc_1116_new_n72_), .B(_abc_1116_new_n73_), .C(_abc_1116_new_n54_), .D(_abc_1116_new_n70_), .Y(_abc_1116_new_n74_));
OAI22X1 OAI22X1_3 ( .A(TEST), .B(_abc_1116_new_n49_), .C(_abc_1116_new_n115_), .D(_abc_1116_new_n114_), .Y(_abc_1116_new_n116_));
OAI22X1 OAI22X1_4 ( .A(VOTO1_REG), .B(_abc_1116_new_n153_), .C(_abc_1116_new_n150_), .D(_abc_1116_new_n154_), .Y(_abc_1116_new_n155_));
OAI22X1 OAI22X1_5 ( .A(VOTO2_REG), .B(_abc_1116_new_n153_), .C(_abc_1116_new_n182_), .D(_abc_1116_new_n154_), .Y(_abc_1116_new_n183_));
OR2X2 OR2X2_1 ( .A(_abc_1116_new_n111_), .B(_abc_1116_new_n113_), .Y(_abc_1116_new_n114_));
OR2X2 OR2X2_2 ( .A(_abc_1116_new_n140_), .B(_abc_1116_new_n72_), .Y(_abc_1116_new_n157_));
OR2X2 OR2X2_3 ( .A(_abc_1116_new_n190_), .B(_abc_1116_new_n110_), .Y(_abc_1116_new_n196_));


endmodule