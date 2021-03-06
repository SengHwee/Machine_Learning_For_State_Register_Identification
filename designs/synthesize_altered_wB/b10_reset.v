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
wire _abc_1116_new_n118_; 
wire _abc_1116_new_n119_; 
wire _abc_1116_new_n120_; 
wire _abc_1116_new_n121_; 
wire _abc_1116_new_n122_; 
wire _abc_1116_new_n123_; 
wire _abc_1116_new_n125_; 
wire _abc_1116_new_n126_; 
wire _abc_1116_new_n127_; 
wire _abc_1116_new_n128_; 
wire _abc_1116_new_n129_; 
wire _abc_1116_new_n130_; 
wire _abc_1116_new_n131_; 
wire _abc_1116_new_n132_; 
wire _abc_1116_new_n133_; 
wire _abc_1116_new_n134_; 
wire _abc_1116_new_n135_; 
wire _abc_1116_new_n136_; 
wire _abc_1116_new_n137_; 
wire _abc_1116_new_n139_; 
wire _abc_1116_new_n140_; 
wire _abc_1116_new_n141_; 
wire _abc_1116_new_n142_; 
wire _abc_1116_new_n143_; 
wire _abc_1116_new_n144_; 
wire _abc_1116_new_n145_; 
wire _abc_1116_new_n146_; 
wire _abc_1116_new_n147_; 
wire _abc_1116_new_n149_; 
wire _abc_1116_new_n150_; 
wire _abc_1116_new_n151_; 
wire _abc_1116_new_n152_; 
wire _abc_1116_new_n153_; 
wire _abc_1116_new_n154_; 
wire _abc_1116_new_n155_; 
wire _abc_1116_new_n156_; 
wire _abc_1116_new_n158_; 
wire _abc_1116_new_n159_; 
wire _abc_1116_new_n160_; 
wire _abc_1116_new_n162_; 
wire _abc_1116_new_n163_; 
wire _abc_1116_new_n164_; 
wire _abc_1116_new_n166_; 
wire _abc_1116_new_n167_; 
wire _abc_1116_new_n168_; 
wire _abc_1116_new_n170_; 
wire _abc_1116_new_n171_; 
wire _abc_1116_new_n172_; 
wire _abc_1116_new_n174_; 
wire _abc_1116_new_n175_; 
wire _abc_1116_new_n176_; 
wire _abc_1116_new_n177_; 
wire _abc_1116_new_n178_; 
wire _abc_1116_new_n179_; 
wire _abc_1116_new_n180_; 
wire _abc_1116_new_n181_; 
wire _abc_1116_new_n182_; 
wire _abc_1116_new_n183_; 
wire _abc_1116_new_n184_; 
wire _abc_1116_new_n185_; 
wire _abc_1116_new_n186_; 
wire _abc_1116_new_n187_; 
wire _abc_1116_new_n188_; 
wire _abc_1116_new_n189_; 
wire _abc_1116_new_n190_; 
wire _abc_1116_new_n191_; 
wire _abc_1116_new_n192_; 
wire _abc_1116_new_n193_; 
wire _abc_1116_new_n194_; 
wire _abc_1116_new_n195_; 
wire _abc_1116_new_n196_; 
wire _abc_1116_new_n197_; 
wire _abc_1116_new_n198_; 
wire _abc_1116_new_n199_; 
wire _abc_1116_new_n200_; 
wire _abc_1116_new_n201_; 
wire _abc_1116_new_n202_; 
wire _abc_1116_new_n204_; 
wire _abc_1116_new_n205_; 
wire _abc_1116_new_n206_; 
wire _abc_1116_new_n207_; 
wire _abc_1116_new_n208_; 
wire _abc_1116_new_n209_; 
wire _abc_1116_new_n210_; 
wire _abc_1116_new_n211_; 
wire _abc_1116_new_n212_; 
wire _abc_1116_new_n213_; 
wire _abc_1116_new_n214_; 
wire _abc_1116_new_n215_; 
wire _abc_1116_new_n216_; 
wire _abc_1116_new_n217_; 
wire _abc_1116_new_n218_; 
wire _abc_1116_new_n219_; 
wire _abc_1116_new_n220_; 
wire _abc_1116_new_n221_; 
wire _abc_1116_new_n222_; 
wire _abc_1116_new_n224_; 
wire _abc_1116_new_n225_; 
wire _abc_1116_new_n226_; 
wire _abc_1116_new_n227_; 
wire _abc_1116_new_n228_; 
wire _abc_1116_new_n230_; 
wire _abc_1116_new_n231_; 
wire _abc_1116_new_n232_; 
wire _abc_1116_new_n233_; 
wire _abc_1116_new_n234_; 
wire _abc_1116_new_n235_; 
wire _abc_1116_new_n236_; 
wire _abc_1116_new_n237_; 
wire _abc_1116_new_n238_; 
wire _abc_1116_new_n239_; 
wire _abc_1116_new_n240_; 
wire _abc_1116_new_n241_; 
wire _abc_1116_new_n242_; 
wire _abc_1116_new_n244_; 
wire _abc_1116_new_n245_; 
wire _abc_1116_new_n246_; 
wire _abc_1116_new_n248_; 
wire _abc_1116_new_n249_; 
wire _abc_1116_new_n250_; 
wire _abc_1116_new_n251_; 
wire _abc_1116_new_n252_; 
wire _abc_1116_new_n253_; 
wire _abc_1116_new_n254_; 
wire _abc_1116_new_n255_; 
wire _abc_1116_new_n256_; 
wire _abc_1116_new_n257_; 
wire _abc_1116_new_n258_; 
wire _abc_1116_new_n259_; 
wire _abc_1116_new_n260_; 
wire _abc_1116_new_n261_; 
wire _abc_1116_new_n262_; 
wire _abc_1116_new_n264_; 
wire _abc_1116_new_n265_; 
wire _abc_1116_new_n266_; 
wire _abc_1116_new_n267_; 
wire _abc_1116_new_n268_; 
wire _abc_1116_new_n269_; 
wire _abc_1116_new_n270_; 
wire _abc_1116_new_n271_; 
wire _abc_1116_new_n272_; 
wire _abc_1116_new_n273_; 
wire _abc_1116_new_n274_; 
wire _abc_1116_new_n275_; 
wire _abc_1116_new_n276_; 
wire _abc_1116_new_n277_; 
wire _abc_1116_new_n279_; 
wire _abc_1116_new_n280_; 
wire _abc_1116_new_n47_; 
wire _abc_1116_new_n48_; 
wire _abc_1116_new_n49_; 
wire _abc_1116_new_n50_; 
wire _abc_1116_new_n51_; 
wire _abc_1116_new_n52_; 
wire _abc_1116_new_n53_; 
wire _abc_1116_new_n54_; 
wire _abc_1116_new_n55_; 
wire _abc_1116_new_n56_; 
wire _abc_1116_new_n57_; 
wire _abc_1116_new_n58_; 
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
wire _abc_1116_new_n89_; 
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
wire _auto_iopadmap_cc_368_execute_1352; 
wire _auto_iopadmap_cc_368_execute_1354; 
wire _auto_iopadmap_cc_368_execute_1356; 
wire _auto_iopadmap_cc_368_execute_1358; 
wire _auto_iopadmap_cc_368_execute_1360; 
wire _auto_iopadmap_cc_368_execute_1362; 
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
AND2X2 AND2X2_1 ( .A(_abc_1116_new_n47_), .B(_abc_1116_new_n48_), .Y(_abc_1116_new_n49_));
AND2X2 AND2X2_10 ( .A(STATO_REG_1_), .B(STATO_REG_0_), .Y(_abc_1116_new_n75_));
AND2X2 AND2X2_100 ( .A(_abc_1116_new_n266_), .B(RTR), .Y(_abc_1116_new_n267_));
AND2X2 AND2X2_101 ( .A(_abc_1116_new_n269_), .B(_auto_iopadmap_cc_368_execute_1354), .Y(_abc_1116_new_n270_));
AND2X2 AND2X2_102 ( .A(STATO_REG_2_), .B(STATO_REG_3_), .Y(_abc_1116_new_n271_));
AND2X2 AND2X2_103 ( .A(_abc_1116_new_n51_), .B(_abc_1116_new_n271_), .Y(_abc_1116_new_n272_));
AND2X2 AND2X2_104 ( .A(_abc_1116_new_n96_), .B(RTR), .Y(_abc_1116_new_n274_));
AND2X2 AND2X2_105 ( .A(_abc_1116_new_n274_), .B(_abc_1116_new_n49_), .Y(_abc_1116_new_n275_));
AND2X2 AND2X2_106 ( .A(_abc_1116_new_n65_), .B(_auto_iopadmap_cc_368_execute_1352), .Y(_abc_1116_new_n279_));
AND2X2 AND2X2_11 ( .A(_abc_1116_new_n74_), .B(_abc_1116_new_n78_), .Y(_abc_1116_new_n79_));
AND2X2 AND2X2_12 ( .A(_abc_1116_new_n79_), .B(_abc_1116_new_n65_), .Y(_abc_1116_new_n80_));
AND2X2 AND2X2_13 ( .A(V_IN_1_), .B(V_IN_3_), .Y(_abc_1116_new_n82_));
AND2X2 AND2X2_14 ( .A(_abc_1116_new_n82_), .B(V_IN_2_), .Y(_abc_1116_new_n83_));
AND2X2 AND2X2_15 ( .A(_abc_1116_new_n83_), .B(V_IN_0_), .Y(_abc_1116_new_n84_));
AND2X2 AND2X2_16 ( .A(_abc_1116_new_n88_), .B(_abc_1116_new_n81_), .Y(_abc_1116_new_n89_));
AND2X2 AND2X2_17 ( .A(_abc_1116_new_n89_), .B(_abc_1116_new_n87_), .Y(_abc_1116_new_n90_));
AND2X2 AND2X2_18 ( .A(_abc_1116_new_n47_), .B(START), .Y(_abc_1116_new_n93_));
AND2X2 AND2X2_19 ( .A(_abc_1116_new_n93_), .B(_abc_1116_new_n48_), .Y(_abc_1116_new_n94_));
AND2X2 AND2X2_2 ( .A(_abc_1116_new_n51_), .B(_abc_1116_new_n49_), .Y(_abc_1116_new_n52_));
AND2X2 AND2X2_20 ( .A(_abc_1116_new_n94_), .B(STATO_REG_0_), .Y(_abc_1116_new_n95_));
AND2X2 AND2X2_21 ( .A(_abc_1116_new_n61_), .B(STATO_REG_0_), .Y(_abc_1116_new_n96_));
AND2X2 AND2X2_22 ( .A(_abc_1116_new_n60_), .B(STATO_REG_2_), .Y(_abc_1116_new_n97_));
AND2X2 AND2X2_23 ( .A(_abc_1116_new_n96_), .B(_abc_1116_new_n97_), .Y(_abc_1116_new_n98_));
AND2X2 AND2X2_24 ( .A(_abc_1116_new_n92_), .B(_abc_1116_new_n100_), .Y(_abc_1116_new_n101_));
AND2X2 AND2X2_25 ( .A(_abc_1116_new_n101_), .B(_abc_1116_new_n80_), .Y(_abc_1116_new_n102_));
AND2X2 AND2X2_26 ( .A(_abc_1116_new_n102_), .B(STATO_REG_1_), .Y(_abc_1116_new_n103_));
AND2X2 AND2X2_27 ( .A(_abc_1116_new_n81_), .B(_abc_1116_new_n48_), .Y(_abc_1116_new_n104_));
AND2X2 AND2X2_28 ( .A(_abc_1116_new_n104_), .B(STATO_REG_1_), .Y(_abc_1116_new_n105_));
AND2X2 AND2X2_29 ( .A(_abc_1116_new_n110_), .B(_abc_1116_new_n66_), .Y(_abc_1116_new_n111_));
AND2X2 AND2X2_3 ( .A(_abc_1116_new_n54_), .B(SIGN_REG_3_), .Y(_abc_1116_new_n55_));
AND2X2 AND2X2_30 ( .A(_abc_1116_new_n116_), .B(STATO_REG_3_), .Y(_abc_1116_new_n117_));
AND2X2 AND2X2_31 ( .A(_abc_1116_new_n117_), .B(_abc_1116_new_n85_), .Y(_abc_1116_new_n118_));
AND2X2 AND2X2_32 ( .A(_abc_1116_new_n120_), .B(_abc_1116_new_n121_), .Y(_abc_1116_new_n122_));
AND2X2 AND2X2_33 ( .A(_abc_1116_new_n125_), .B(STATO_REG_3_), .Y(_abc_1116_new_n126_));
AND2X2 AND2X2_34 ( .A(_abc_1116_new_n128_), .B(VOTO2_REG), .Y(_abc_1116_new_n129_));
AND2X2 AND2X2_35 ( .A(_abc_1116_new_n130_), .B(VOTO1_REG), .Y(_abc_1116_new_n131_));
AND2X2 AND2X2_36 ( .A(_abc_1116_new_n129_), .B(_abc_1116_new_n131_), .Y(_abc_1116_new_n132_));
AND2X2 AND2X2_37 ( .A(_abc_1116_new_n110_), .B(_abc_1116_new_n61_), .Y(_abc_1116_new_n133_));
AND2X2 AND2X2_38 ( .A(_abc_1116_new_n133_), .B(_abc_1116_new_n132_), .Y(_abc_1116_new_n134_));
AND2X2 AND2X2_39 ( .A(_abc_1116_new_n120_), .B(_abc_1116_new_n135_), .Y(_abc_1116_new_n136_));
AND2X2 AND2X2_4 ( .A(STATO_REG_0_), .B(STATO_REG_3_), .Y(_abc_1116_new_n57_));
AND2X2 AND2X2_40 ( .A(_abc_1116_new_n139_), .B(_abc_1116_new_n81_), .Y(_abc_1116_new_n140_));
AND2X2 AND2X2_41 ( .A(_abc_1116_new_n142_), .B(STATO_REG_2_), .Y(_abc_1116_new_n143_));
AND2X2 AND2X2_42 ( .A(_abc_1116_new_n145_), .B(STATO_REG_1_), .Y(_abc_1116_new_n146_));
AND2X2 AND2X2_43 ( .A(_abc_1116_new_n149_), .B(_abc_1116_new_n51_), .Y(_abc_1116_new_n150_));
AND2X2 AND2X2_44 ( .A(_abc_1116_new_n49_), .B(_abc_1116_new_n81_), .Y(_abc_1116_new_n151_));
AND2X2 AND2X2_45 ( .A(_abc_1116_new_n120_), .B(_abc_1116_new_n153_), .Y(_abc_1116_new_n154_));
AND2X2 AND2X2_46 ( .A(_abc_1116_new_n102_), .B(STATO_REG_0_), .Y(_abc_1116_new_n155_));
AND2X2 AND2X2_47 ( .A(_abc_1116_new_n68_), .B(_auto_iopadmap_cc_368_execute_1362), .Y(_abc_1116_new_n158_));
AND2X2 AND2X2_48 ( .A(_abc_1116_new_n111_), .B(VOTO3_REG), .Y(_abc_1116_new_n160_));
AND2X2 AND2X2_49 ( .A(_abc_1116_new_n68_), .B(_auto_iopadmap_cc_368_execute_1360), .Y(_abc_1116_new_n162_));
AND2X2 AND2X2_5 ( .A(_abc_1116_new_n64_), .B(_abc_1116_new_n53_), .Y(_abc_1116_new_n65_));
AND2X2 AND2X2_50 ( .A(_abc_1116_new_n111_), .B(VOTO2_REG), .Y(_abc_1116_new_n163_));
AND2X2 AND2X2_51 ( .A(_abc_1116_new_n68_), .B(_auto_iopadmap_cc_368_execute_1358), .Y(_abc_1116_new_n166_));
AND2X2 AND2X2_52 ( .A(_abc_1116_new_n111_), .B(VOTO1_REG), .Y(_abc_1116_new_n167_));
AND2X2 AND2X2_53 ( .A(_abc_1116_new_n68_), .B(_auto_iopadmap_cc_368_execute_1356), .Y(_abc_1116_new_n170_));
AND2X2 AND2X2_54 ( .A(_abc_1116_new_n111_), .B(VOTO0_REG), .Y(_abc_1116_new_n172_));
AND2X2 AND2X2_55 ( .A(_abc_1116_new_n65_), .B(_abc_1116_new_n176_), .Y(_abc_1116_new_n177_));
AND2X2 AND2X2_56 ( .A(_abc_1116_new_n93_), .B(_abc_1116_new_n96_), .Y(_abc_1116_new_n178_));
AND2X2 AND2X2_57 ( .A(_abc_1116_new_n177_), .B(_abc_1116_new_n179_), .Y(_abc_1116_new_n180_));
AND2X2 AND2X2_58 ( .A(_abc_1116_new_n104_), .B(_abc_1116_new_n93_), .Y(_abc_1116_new_n182_));
AND2X2 AND2X2_59 ( .A(_abc_1116_new_n182_), .B(_abc_1116_new_n181_), .Y(_abc_1116_new_n183_));
AND2X2 AND2X2_6 ( .A(_abc_1116_new_n61_), .B(RTR), .Y(_abc_1116_new_n66_));
AND2X2 AND2X2_60 ( .A(_abc_1116_new_n184_), .B(G_BUTTON), .Y(_abc_1116_new_n185_));
AND2X2 AND2X2_61 ( .A(_abc_1116_new_n182_), .B(_abc_1116_new_n185_), .Y(_abc_1116_new_n186_));
AND2X2 AND2X2_62 ( .A(_abc_1116_new_n180_), .B(_abc_1116_new_n188_), .Y(_abc_1116_new_n189_));
AND2X2 AND2X2_63 ( .A(STATO_REG_1_), .B(KEY), .Y(_abc_1116_new_n192_));
AND2X2 AND2X2_64 ( .A(_abc_1116_new_n192_), .B(_abc_1116_new_n47_), .Y(_abc_1116_new_n193_));
AND2X2 AND2X2_65 ( .A(_abc_1116_new_n193_), .B(_abc_1116_new_n191_), .Y(_abc_1116_new_n194_));
AND2X2 AND2X2_66 ( .A(_abc_1116_new_n105_), .B(STATO_REG_2_), .Y(_abc_1116_new_n196_));
AND2X2 AND2X2_67 ( .A(_abc_1116_new_n197_), .B(V_IN_1_), .Y(_abc_1116_new_n198_));
AND2X2 AND2X2_68 ( .A(_abc_1116_new_n190_), .B(_abc_1116_new_n199_), .Y(_abc_1116_new_n200_));
AND2X2 AND2X2_69 ( .A(_abc_1116_new_n189_), .B(VOTO1_REG), .Y(_abc_1116_new_n201_));
AND2X2 AND2X2_7 ( .A(_abc_1116_new_n47_), .B(STATO_REG_1_), .Y(_abc_1116_new_n69_));
AND2X2 AND2X2_70 ( .A(_abc_1116_new_n181_), .B(START), .Y(_abc_1116_new_n204_));
AND2X2 AND2X2_71 ( .A(_abc_1116_new_n205_), .B(_abc_1116_new_n49_), .Y(_abc_1116_new_n206_));
AND2X2 AND2X2_72 ( .A(_abc_1116_new_n180_), .B(_abc_1116_new_n207_), .Y(_abc_1116_new_n208_));
AND2X2 AND2X2_73 ( .A(_abc_1116_new_n197_), .B(V_IN_3_), .Y(_abc_1116_new_n210_));
AND2X2 AND2X2_74 ( .A(_abc_1116_new_n211_), .B(VOTO0_REG), .Y(_abc_1116_new_n212_));
AND2X2 AND2X2_75 ( .A(_abc_1116_new_n214_), .B(VOTO1_REG), .Y(_abc_1116_new_n215_));
AND2X2 AND2X2_76 ( .A(_abc_1116_new_n213_), .B(_abc_1116_new_n191_), .Y(_abc_1116_new_n216_));
AND2X2 AND2X2_77 ( .A(_abc_1116_new_n217_), .B(_abc_1116_new_n75_), .Y(_abc_1116_new_n218_));
AND2X2 AND2X2_78 ( .A(_abc_1116_new_n209_), .B(_abc_1116_new_n219_), .Y(_abc_1116_new_n220_));
AND2X2 AND2X2_79 ( .A(_abc_1116_new_n208_), .B(VOTO3_REG), .Y(_abc_1116_new_n221_));
AND2X2 AND2X2_8 ( .A(_abc_1116_new_n71_), .B(_abc_1116_new_n69_), .Y(_abc_1116_new_n72_));
AND2X2 AND2X2_80 ( .A(_abc_1116_new_n182_), .B(_abc_1116_new_n192_), .Y(_abc_1116_new_n224_));
AND2X2 AND2X2_81 ( .A(_abc_1116_new_n225_), .B(LAST_R_REG), .Y(_abc_1116_new_n226_));
AND2X2 AND2X2_82 ( .A(_abc_1116_new_n224_), .B(R_BUTTON), .Y(_abc_1116_new_n227_));
AND2X2 AND2X2_83 ( .A(_abc_1116_new_n230_), .B(R_BUTTON), .Y(_abc_1116_new_n231_));
AND2X2 AND2X2_84 ( .A(_abc_1116_new_n182_), .B(_abc_1116_new_n231_), .Y(_abc_1116_new_n232_));
AND2X2 AND2X2_85 ( .A(_abc_1116_new_n180_), .B(_abc_1116_new_n234_), .Y(_abc_1116_new_n235_));
AND2X2 AND2X2_86 ( .A(_abc_1116_new_n193_), .B(_abc_1116_new_n211_), .Y(_abc_1116_new_n237_));
AND2X2 AND2X2_87 ( .A(_abc_1116_new_n197_), .B(V_IN_2_), .Y(_abc_1116_new_n238_));
AND2X2 AND2X2_88 ( .A(_abc_1116_new_n236_), .B(_abc_1116_new_n239_), .Y(_abc_1116_new_n240_));
AND2X2 AND2X2_89 ( .A(_abc_1116_new_n235_), .B(VOTO2_REG), .Y(_abc_1116_new_n241_));
AND2X2 AND2X2_9 ( .A(_abc_1116_new_n73_), .B(_abc_1116_new_n68_), .Y(_abc_1116_new_n74_));
AND2X2 AND2X2_90 ( .A(_abc_1116_new_n245_), .B(_abc_1116_new_n244_), .Y(_abc_1116_new_n246_));
AND2X2 AND2X2_91 ( .A(_abc_1116_new_n249_), .B(_abc_1116_new_n248_), .Y(_abc_1116_new_n250_));
AND2X2 AND2X2_92 ( .A(_abc_1116_new_n177_), .B(_abc_1116_new_n250_), .Y(_abc_1116_new_n251_));
AND2X2 AND2X2_93 ( .A(_abc_1116_new_n197_), .B(V_IN_0_), .Y(_abc_1116_new_n253_));
AND2X2 AND2X2_94 ( .A(_abc_1116_new_n151_), .B(_abc_1116_new_n192_), .Y(_abc_1116_new_n254_));
AND2X2 AND2X2_95 ( .A(_abc_1116_new_n255_), .B(STATO_REG_3_), .Y(_abc_1116_new_n256_));
AND2X2 AND2X2_96 ( .A(_abc_1116_new_n256_), .B(STATO_REG_1_), .Y(_abc_1116_new_n257_));
AND2X2 AND2X2_97 ( .A(_abc_1116_new_n252_), .B(_abc_1116_new_n259_), .Y(_abc_1116_new_n260_));
AND2X2 AND2X2_98 ( .A(_abc_1116_new_n251_), .B(VOTO0_REG), .Y(_abc_1116_new_n261_));
AND2X2 AND2X2_99 ( .A(_abc_1116_new_n264_), .B(_abc_1116_new_n76_), .Y(_abc_1116_new_n265_));
BUFX2 BUFX2_1 ( .A(_auto_iopadmap_cc_368_execute_1352), .Y(CTR_REG));
BUFX2 BUFX2_2 ( .A(_auto_iopadmap_cc_368_execute_1354), .Y(CTS_REG));
BUFX2 BUFX2_3 ( .A(_auto_iopadmap_cc_368_execute_1356), .Y(V_OUT_REG_0_));
BUFX2 BUFX2_4 ( .A(_auto_iopadmap_cc_368_execute_1358), .Y(V_OUT_REG_1_));
BUFX2 BUFX2_5 ( .A(_auto_iopadmap_cc_368_execute_1360), .Y(V_OUT_REG_2_));
BUFX2 BUFX2_6 ( .A(_auto_iopadmap_cc_368_execute_1362), .Y(V_OUT_REG_3_));
BUFX4 BUFX4_1 ( .A(clock), .Y(clock_bF_buf3));
BUFX4 BUFX4_2 ( .A(clock), .Y(clock_bF_buf2));
BUFX4 BUFX4_3 ( .A(clock), .Y(clock_bF_buf1));
BUFX4 BUFX4_4 ( .A(clock), .Y(clock_bF_buf0));
DFFPOSX1 DFFPOSX1_1 ( .CLK(clock_bF_buf3), .D(n105), .Q(_auto_iopadmap_cc_368_execute_1354));
DFFPOSX1 DFFPOSX1_10 ( .CLK(clock_bF_buf2), .D(n55), .Q(STATO_REG_1_));
DFFPOSX1 DFFPOSX1_11 ( .CLK(clock_bF_buf1), .D(n60), .Q(STATO_REG_0_));
DFFPOSX1 DFFPOSX1_12 ( .CLK(clock_bF_buf0), .D(n81), .Q(SIGN_REG_3_));
DFFPOSX1 DFFPOSX1_13 ( .CLK(clock_bF_buf3), .D(n86), .Q(VOTO1_REG));
DFFPOSX1 DFFPOSX1_14 ( .CLK(clock_bF_buf2), .D(n95), .Q(VOTO3_REG));
DFFPOSX1 DFFPOSX1_15 ( .CLK(clock_bF_buf1), .D(n100), .Q(LAST_R_REG));
DFFPOSX1 DFFPOSX1_16 ( .CLK(clock_bF_buf0), .D(n109), .Q(VOTO2_REG));
DFFPOSX1 DFFPOSX1_17 ( .CLK(clock_bF_buf3), .D(n114), .Q(LAST_G_REG));
DFFPOSX1 DFFPOSX1_2 ( .CLK(clock_bF_buf2), .D(n65), .Q(_auto_iopadmap_cc_368_execute_1362));
DFFPOSX1 DFFPOSX1_3 ( .CLK(clock_bF_buf1), .D(n77), .Q(_auto_iopadmap_cc_368_execute_1356));
DFFPOSX1 DFFPOSX1_4 ( .CLK(clock_bF_buf0), .D(n69), .Q(_auto_iopadmap_cc_368_execute_1360));
DFFPOSX1 DFFPOSX1_5 ( .CLK(clock_bF_buf3), .D(n73), .Q(_auto_iopadmap_cc_368_execute_1358));
DFFPOSX1 DFFPOSX1_6 ( .CLK(clock_bF_buf2), .D(n91), .Q(_auto_iopadmap_cc_368_execute_1352));
DFFPOSX1 DFFPOSX1_7 ( .CLK(clock_bF_buf1), .D(n40), .Q(VOTO0_REG));
DFFPOSX1 DFFPOSX1_8 ( .CLK(clock_bF_buf0), .D(n45), .Q(STATO_REG_3_));
DFFPOSX1 DFFPOSX1_9 ( .CLK(clock_bF_buf3), .D(n50), .Q(STATO_REG_2_));
INVX1 INVX1_1 ( .A(STATO_REG_3_), .Y(_abc_1116_new_n48_));
INVX1 INVX1_10 ( .A(_abc_1116_new_n99_), .Y(_abc_1116_new_n100_));
INVX1 INVX1_11 ( .A(_abc_1116_new_n63_), .Y(_abc_1116_new_n106_));
INVX1 INVX1_12 ( .A(_abc_1116_new_n65_), .Y(_abc_1116_new_n109_));
INVX1 INVX1_13 ( .A(_abc_1116_new_n62_), .Y(_abc_1116_new_n110_));
INVX1 INVX1_14 ( .A(_abc_1116_new_n78_), .Y(_abc_1116_new_n113_));
INVX1 INVX1_15 ( .A(_abc_1116_new_n90_), .Y(_abc_1116_new_n116_));
INVX1 INVX1_16 ( .A(_abc_1116_new_n54_), .Y(_abc_1116_new_n127_));
INVX1 INVX1_17 ( .A(VOTO0_REG), .Y(_abc_1116_new_n128_));
INVX1 INVX1_18 ( .A(VOTO3_REG), .Y(_abc_1116_new_n130_));
INVX1 INVX1_19 ( .A(_abc_1116_new_n132_), .Y(_abc_1116_new_n139_));
INVX1 INVX1_2 ( .A(_abc_1116_new_n50_), .Y(_abc_1116_new_n51_));
INVX1 INVX1_20 ( .A(_abc_1116_new_n144_), .Y(_abc_1116_new_n145_));
INVX1 INVX1_21 ( .A(_abc_1116_new_n57_), .Y(_abc_1116_new_n174_));
INVX1 INVX1_22 ( .A(_abc_1116_new_n178_), .Y(_abc_1116_new_n179_));
INVX1 INVX1_23 ( .A(KEY), .Y(_abc_1116_new_n181_));
INVX1 INVX1_24 ( .A(LAST_G_REG), .Y(_abc_1116_new_n184_));
INVX1 INVX1_25 ( .A(_abc_1116_new_n187_), .Y(_abc_1116_new_n188_));
INVX1 INVX1_26 ( .A(_abc_1116_new_n189_), .Y(_abc_1116_new_n190_));
INVX1 INVX1_27 ( .A(VOTO1_REG), .Y(_abc_1116_new_n191_));
INVX1 INVX1_28 ( .A(_abc_1116_new_n176_), .Y(_abc_1116_new_n195_));
INVX1 INVX1_29 ( .A(_abc_1116_new_n206_), .Y(_abc_1116_new_n207_));
INVX1 INVX1_3 ( .A(_abc_1116_new_n52_), .Y(_abc_1116_new_n53_));
INVX1 INVX1_30 ( .A(_abc_1116_new_n208_), .Y(_abc_1116_new_n209_));
INVX1 INVX1_31 ( .A(VOTO2_REG), .Y(_abc_1116_new_n211_));
INVX1 INVX1_32 ( .A(_abc_1116_new_n213_), .Y(_abc_1116_new_n214_));
INVX1 INVX1_33 ( .A(_abc_1116_new_n224_), .Y(_abc_1116_new_n225_));
INVX1 INVX1_34 ( .A(LAST_R_REG), .Y(_abc_1116_new_n230_));
INVX1 INVX1_35 ( .A(_abc_1116_new_n233_), .Y(_abc_1116_new_n234_));
INVX1 INVX1_36 ( .A(_abc_1116_new_n235_), .Y(_abc_1116_new_n236_));
INVX1 INVX1_37 ( .A(_abc_1116_new_n94_), .Y(_abc_1116_new_n248_));
INVX1 INVX1_38 ( .A(_abc_1116_new_n251_), .Y(_abc_1116_new_n252_));
INVX1 INVX1_39 ( .A(SIGN_REG_3_), .Y(_abc_1116_new_n255_));
INVX1 INVX1_4 ( .A(RTS), .Y(_abc_1116_new_n60_));
INVX1 INVX1_40 ( .A(_abc_1116_new_n104_), .Y(_abc_1116_new_n266_));
INVX1 INVX1_5 ( .A(_abc_1116_new_n66_), .Y(_abc_1116_new_n67_));
INVX1 INVX1_6 ( .A(_abc_1116_new_n70_), .Y(_abc_1116_new_n71_));
INVX1 INVX1_7 ( .A(_abc_1116_new_n72_), .Y(_abc_1116_new_n73_));
INVX1 INVX1_8 ( .A(_abc_1116_new_n75_), .Y(_abc_1116_new_n76_));
INVX1 INVX1_9 ( .A(_abc_1116_new_n85_), .Y(_abc_1116_new_n86_));
INVX2 INVX2_1 ( .A(STATO_REG_2_), .Y(_abc_1116_new_n47_));
INVX2 INVX2_2 ( .A(STATO_REG_1_), .Y(_abc_1116_new_n61_));
INVX2 INVX2_3 ( .A(STATO_REG_0_), .Y(_abc_1116_new_n81_));
INVX4 INVX4_1 ( .A(nRESET_G), .Y(_abc_1116_new_n56_));
OR2X2 OR2X2_1 ( .A(STATO_REG_1_), .B(STATO_REG_0_), .Y(_abc_1116_new_n50_));
OR2X2 OR2X2_10 ( .A(_abc_1116_new_n47_), .B(RTR), .Y(_abc_1116_new_n77_));
OR2X2 OR2X2_11 ( .A(_abc_1116_new_n77_), .B(_abc_1116_new_n76_), .Y(_abc_1116_new_n78_));
OR2X2 OR2X2_12 ( .A(_abc_1116_new_n84_), .B(_abc_1116_new_n81_), .Y(_abc_1116_new_n85_));
OR2X2 OR2X2_13 ( .A(_abc_1116_new_n61_), .B(STATO_REG_2_), .Y(_abc_1116_new_n87_));
OR2X2 OR2X2_14 ( .A(STATO_REG_1_), .B(RTR), .Y(_abc_1116_new_n88_));
OR2X2 OR2X2_15 ( .A(_abc_1116_new_n90_), .B(_abc_1116_new_n48_), .Y(_abc_1116_new_n91_));
OR2X2 OR2X2_16 ( .A(_abc_1116_new_n86_), .B(_abc_1116_new_n91_), .Y(_abc_1116_new_n92_));
OR2X2 OR2X2_17 ( .A(_abc_1116_new_n95_), .B(_abc_1116_new_n98_), .Y(_abc_1116_new_n99_));
OR2X2 OR2X2_18 ( .A(_abc_1116_new_n106_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n107_));
OR2X2 OR2X2_19 ( .A(_abc_1116_new_n107_), .B(_abc_1116_new_n105_), .Y(_abc_1116_new_n108_));
OR2X2 OR2X2_2 ( .A(_abc_1116_new_n53_), .B(TEST), .Y(_abc_1116_new_n54_));
OR2X2 OR2X2_20 ( .A(_abc_1116_new_n111_), .B(_abc_1116_new_n72_), .Y(_abc_1116_new_n112_));
OR2X2 OR2X2_21 ( .A(_abc_1116_new_n112_), .B(_abc_1116_new_n113_), .Y(_abc_1116_new_n114_));
OR2X2 OR2X2_22 ( .A(_abc_1116_new_n114_), .B(_abc_1116_new_n109_), .Y(_abc_1116_new_n115_));
OR2X2 OR2X2_23 ( .A(_abc_1116_new_n118_), .B(_abc_1116_new_n99_), .Y(_abc_1116_new_n119_));
OR2X2 OR2X2_24 ( .A(_abc_1116_new_n119_), .B(_abc_1116_new_n115_), .Y(_abc_1116_new_n120_));
OR2X2 OR2X2_25 ( .A(_abc_1116_new_n96_), .B(_abc_1116_new_n57_), .Y(_abc_1116_new_n121_));
OR2X2 OR2X2_26 ( .A(_abc_1116_new_n122_), .B(_abc_1116_new_n108_), .Y(_abc_1116_new_n123_));
OR2X2 OR2X2_27 ( .A(_abc_1116_new_n123_), .B(_abc_1116_new_n103_), .Y(n55));
OR2X2 OR2X2_28 ( .A(_abc_1116_new_n102_), .B(STATO_REG_0_), .Y(_abc_1116_new_n125_));
OR2X2 OR2X2_29 ( .A(_abc_1116_new_n127_), .B(_abc_1116_new_n134_), .Y(_abc_1116_new_n135_));
OR2X2 OR2X2_3 ( .A(_abc_1116_new_n57_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n58_));
OR2X2 OR2X2_30 ( .A(_abc_1116_new_n136_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n137_));
OR2X2 OR2X2_31 ( .A(_abc_1116_new_n137_), .B(_abc_1116_new_n126_), .Y(n45));
OR2X2 OR2X2_32 ( .A(_abc_1116_new_n140_), .B(_abc_1116_new_n96_), .Y(_abc_1116_new_n141_));
OR2X2 OR2X2_33 ( .A(_abc_1116_new_n102_), .B(_abc_1116_new_n141_), .Y(_abc_1116_new_n142_));
OR2X2 OR2X2_34 ( .A(_abc_1116_new_n104_), .B(_abc_1116_new_n57_), .Y(_abc_1116_new_n144_));
OR2X2 OR2X2_35 ( .A(_abc_1116_new_n107_), .B(_abc_1116_new_n146_), .Y(_abc_1116_new_n147_));
OR2X2 OR2X2_36 ( .A(_abc_1116_new_n143_), .B(_abc_1116_new_n147_), .Y(n50));
OR2X2 OR2X2_37 ( .A(_abc_1116_new_n139_), .B(STATO_REG_3_), .Y(_abc_1116_new_n149_));
OR2X2 OR2X2_38 ( .A(_abc_1116_new_n106_), .B(_abc_1116_new_n151_), .Y(_abc_1116_new_n152_));
OR2X2 OR2X2_39 ( .A(_abc_1116_new_n150_), .B(_abc_1116_new_n152_), .Y(_abc_1116_new_n153_));
OR2X2 OR2X2_4 ( .A(_abc_1116_new_n55_), .B(_abc_1116_new_n58_), .Y(n81));
OR2X2 OR2X2_40 ( .A(_abc_1116_new_n155_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n156_));
OR2X2 OR2X2_41 ( .A(_abc_1116_new_n156_), .B(_abc_1116_new_n154_), .Y(n60));
OR2X2 OR2X2_42 ( .A(_abc_1116_new_n158_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n159_));
OR2X2 OR2X2_43 ( .A(_abc_1116_new_n159_), .B(_abc_1116_new_n160_), .Y(n65));
OR2X2 OR2X2_44 ( .A(_abc_1116_new_n163_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n164_));
OR2X2 OR2X2_45 ( .A(_abc_1116_new_n164_), .B(_abc_1116_new_n162_), .Y(n69));
OR2X2 OR2X2_46 ( .A(_abc_1116_new_n167_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n168_));
OR2X2 OR2X2_47 ( .A(_abc_1116_new_n168_), .B(_abc_1116_new_n166_), .Y(n73));
OR2X2 OR2X2_48 ( .A(_abc_1116_new_n170_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n171_));
OR2X2 OR2X2_49 ( .A(_abc_1116_new_n171_), .B(_abc_1116_new_n172_), .Y(n77));
OR2X2 OR2X2_5 ( .A(_abc_1116_new_n47_), .B(STATO_REG_0_), .Y(_abc_1116_new_n62_));
OR2X2 OR2X2_50 ( .A(STATO_REG_1_), .B(STATO_REG_2_), .Y(_abc_1116_new_n175_));
OR2X2 OR2X2_51 ( .A(_abc_1116_new_n174_), .B(_abc_1116_new_n175_), .Y(_abc_1116_new_n176_));
OR2X2 OR2X2_52 ( .A(_abc_1116_new_n183_), .B(_abc_1116_new_n186_), .Y(_abc_1116_new_n187_));
OR2X2 OR2X2_53 ( .A(_abc_1116_new_n196_), .B(_abc_1116_new_n195_), .Y(_abc_1116_new_n197_));
OR2X2 OR2X2_54 ( .A(_abc_1116_new_n198_), .B(_abc_1116_new_n194_), .Y(_abc_1116_new_n199_));
OR2X2 OR2X2_55 ( .A(_abc_1116_new_n201_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n202_));
OR2X2 OR2X2_56 ( .A(_abc_1116_new_n202_), .B(_abc_1116_new_n200_), .Y(n86));
OR2X2 OR2X2_57 ( .A(_abc_1116_new_n204_), .B(_abc_1116_new_n75_), .Y(_abc_1116_new_n205_));
OR2X2 OR2X2_58 ( .A(_abc_1116_new_n129_), .B(_abc_1116_new_n212_), .Y(_abc_1116_new_n213_));
OR2X2 OR2X2_59 ( .A(_abc_1116_new_n215_), .B(_abc_1116_new_n216_), .Y(_abc_1116_new_n217_));
OR2X2 OR2X2_6 ( .A(_abc_1116_new_n62_), .B(_abc_1116_new_n61_), .Y(_abc_1116_new_n63_));
OR2X2 OR2X2_60 ( .A(_abc_1116_new_n218_), .B(_abc_1116_new_n210_), .Y(_abc_1116_new_n219_));
OR2X2 OR2X2_61 ( .A(_abc_1116_new_n221_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n222_));
OR2X2 OR2X2_62 ( .A(_abc_1116_new_n222_), .B(_abc_1116_new_n220_), .Y(n95));
OR2X2 OR2X2_63 ( .A(_abc_1116_new_n227_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n228_));
OR2X2 OR2X2_64 ( .A(_abc_1116_new_n228_), .B(_abc_1116_new_n226_), .Y(n100));
OR2X2 OR2X2_65 ( .A(_abc_1116_new_n183_), .B(_abc_1116_new_n232_), .Y(_abc_1116_new_n233_));
OR2X2 OR2X2_66 ( .A(_abc_1116_new_n238_), .B(_abc_1116_new_n237_), .Y(_abc_1116_new_n239_));
OR2X2 OR2X2_67 ( .A(_abc_1116_new_n241_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n242_));
OR2X2 OR2X2_68 ( .A(_abc_1116_new_n242_), .B(_abc_1116_new_n240_), .Y(n109));
OR2X2 OR2X2_69 ( .A(_abc_1116_new_n224_), .B(LAST_G_REG), .Y(_abc_1116_new_n244_));
OR2X2 OR2X2_7 ( .A(_abc_1116_new_n63_), .B(_abc_1116_new_n60_), .Y(_abc_1116_new_n64_));
OR2X2 OR2X2_70 ( .A(_abc_1116_new_n225_), .B(G_BUTTON), .Y(_abc_1116_new_n245_));
OR2X2 OR2X2_71 ( .A(_abc_1116_new_n246_), .B(_abc_1116_new_n56_), .Y(n114));
OR2X2 OR2X2_72 ( .A(_abc_1116_new_n144_), .B(_abc_1116_new_n87_), .Y(_abc_1116_new_n249_));
OR2X2 OR2X2_73 ( .A(_abc_1116_new_n254_), .B(_abc_1116_new_n257_), .Y(_abc_1116_new_n258_));
OR2X2 OR2X2_74 ( .A(_abc_1116_new_n253_), .B(_abc_1116_new_n258_), .Y(_abc_1116_new_n259_));
OR2X2 OR2X2_75 ( .A(_abc_1116_new_n261_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n262_));
OR2X2 OR2X2_76 ( .A(_abc_1116_new_n262_), .B(_abc_1116_new_n260_), .Y(n40));
OR2X2 OR2X2_77 ( .A(_abc_1116_new_n175_), .B(_abc_1116_new_n57_), .Y(_abc_1116_new_n264_));
OR2X2 OR2X2_78 ( .A(_abc_1116_new_n267_), .B(_abc_1116_new_n69_), .Y(_abc_1116_new_n268_));
OR2X2 OR2X2_79 ( .A(_abc_1116_new_n268_), .B(_abc_1116_new_n265_), .Y(_abc_1116_new_n269_));
OR2X2 OR2X2_8 ( .A(_abc_1116_new_n67_), .B(_abc_1116_new_n62_), .Y(_abc_1116_new_n68_));
OR2X2 OR2X2_80 ( .A(_abc_1116_new_n272_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n273_));
OR2X2 OR2X2_81 ( .A(_abc_1116_new_n275_), .B(_abc_1116_new_n111_), .Y(_abc_1116_new_n276_));
OR2X2 OR2X2_82 ( .A(_abc_1116_new_n276_), .B(_abc_1116_new_n273_), .Y(_abc_1116_new_n277_));
OR2X2 OR2X2_83 ( .A(_abc_1116_new_n270_), .B(_abc_1116_new_n277_), .Y(n105));
OR2X2 OR2X2_84 ( .A(_abc_1116_new_n98_), .B(_abc_1116_new_n56_), .Y(_abc_1116_new_n280_));
OR2X2 OR2X2_85 ( .A(_abc_1116_new_n279_), .B(_abc_1116_new_n280_), .Y(n91));
OR2X2 OR2X2_9 ( .A(START), .B(STATO_REG_3_), .Y(_abc_1116_new_n70_));


endmodule