module b09_reset(clock, RESET_G, nRESET_G, X, Y_REG);
  wire D_IN_REG_0_;
  wire D_IN_REG_1_;
  wire D_IN_REG_2_;
  wire D_IN_REG_3_;
  wire D_IN_REG_4_;
  wire D_IN_REG_5_;
  wire D_IN_REG_6_;
  wire D_IN_REG_7_;
  wire D_IN_REG_8_;
  wire D_OUT_REG_0_;
  wire D_OUT_REG_1_;
  wire D_OUT_REG_2_;
  wire D_OUT_REG_3_;
  wire D_OUT_REG_4_;
  wire D_OUT_REG_5_;
  wire D_OUT_REG_6_;
  wire D_OUT_REG_7_;
  wire OLD_REG_0_;
  wire OLD_REG_1_;
  wire OLD_REG_2_;
  wire OLD_REG_3_;
  wire OLD_REG_4_;
  wire OLD_REG_5_;
  wire OLD_REG_6_;
  wire OLD_REG_7_;
  input RESET_G;
  wire STATO_REG_0_;
  wire STATO_REG_1_;
  input X;
  output Y_REG;
  wire _abc_896_n100;
  wire _abc_896_n101;
  wire _abc_896_n102;
  wire _abc_896_n103;
  wire _abc_896_n104;
  wire _abc_896_n105;
  wire _abc_896_n106;
  wire _abc_896_n107_1;
  wire _abc_896_n108;
  wire _abc_896_n109_1;
  wire _abc_896_n110;
  wire _abc_896_n110_bF_buf0;
  wire _abc_896_n110_bF_buf1;
  wire _abc_896_n110_bF_buf2;
  wire _abc_896_n110_bF_buf3;
  wire _abc_896_n111;
  wire _abc_896_n112;
  wire _abc_896_n113_1;
  wire _abc_896_n114;
  wire _abc_896_n115_1;
  wire _abc_896_n116;
  wire _abc_896_n117;
  wire _abc_896_n118;
  wire _abc_896_n119_1;
  wire _abc_896_n121_1;
  wire _abc_896_n123;
  wire _abc_896_n124;
  wire _abc_896_n125_1;
  wire _abc_896_n134;
  wire _abc_896_n135;
  wire _abc_896_n136;
  wire _abc_896_n137_1;
  wire _abc_896_n138;
  wire _abc_896_n139_1;
  wire _abc_896_n140;
  wire _abc_896_n141;
  wire _abc_896_n142;
  wire _abc_896_n143_1;
  wire _abc_896_n144;
  wire _abc_896_n145_1;
  wire _abc_896_n146;
  wire _abc_896_n147;
  wire _abc_896_n148;
  wire _abc_896_n150_1;
  wire _abc_896_n151;
  wire _abc_896_n152;
  wire _abc_896_n153;
  wire _abc_896_n154;
  wire _abc_896_n155;
  wire _abc_896_n156;
  wire _abc_896_n158;
  wire _abc_896_n159;
  wire _abc_896_n160;
  wire _abc_896_n162;
  wire _abc_896_n163;
  wire _abc_896_n164;
  wire _abc_896_n166;
  wire _abc_896_n167;
  wire _abc_896_n168;
  wire _abc_896_n170;
  wire _abc_896_n171;
  wire _abc_896_n172;
  wire _abc_896_n174;
  wire _abc_896_n175;
  wire _abc_896_n176;
  wire _abc_896_n178;
  wire _abc_896_n179;
  wire _abc_896_n180;
  wire _abc_896_n182;
  wire _abc_896_n183;
  wire _abc_896_n184;
  wire _abc_896_n186;
  wire _abc_896_n187;
  wire _abc_896_n188;
  wire _abc_896_n190;
  wire _abc_896_n191;
  wire _abc_896_n192;
  wire _abc_896_n193;
  wire _abc_896_n194;
  wire _abc_896_n195;
  wire _abc_896_n196;
  wire _abc_896_n197;
  wire _abc_896_n198;
  wire _abc_896_n200;
  wire _abc_896_n201;
  wire _abc_896_n202;
  wire _abc_896_n203;
  wire _abc_896_n204;
  wire _abc_896_n206;
  wire _abc_896_n207;
  wire _abc_896_n208;
  wire _abc_896_n209;
  wire _abc_896_n210;
  wire _abc_896_n212;
  wire _abc_896_n213;
  wire _abc_896_n214;
  wire _abc_896_n215;
  wire _abc_896_n216;
  wire _abc_896_n218;
  wire _abc_896_n219;
  wire _abc_896_n220;
  wire _abc_896_n221;
  wire _abc_896_n222;
  wire _abc_896_n224;
  wire _abc_896_n225;
  wire _abc_896_n226;
  wire _abc_896_n227;
  wire _abc_896_n228;
  wire _abc_896_n230;
  wire _abc_896_n231;
  wire _abc_896_n232;
  wire _abc_896_n233;
  wire _abc_896_n234;
  wire _abc_896_n236;
  wire _abc_896_n237;
  wire _abc_896_n238;
  wire _abc_896_n59;
  wire _abc_896_n60;
  wire _abc_896_n61_1;
  wire _abc_896_n62;
  wire _abc_896_n63;
  wire _abc_896_n64_1;
  wire _abc_896_n65;
  wire _abc_896_n66;
  wire _abc_896_n67_1;
  wire _abc_896_n68;
  wire _abc_896_n69;
  wire _abc_896_n70_1;
  wire _abc_896_n71;
  wire _abc_896_n72;
  wire _abc_896_n73;
  wire _abc_896_n74_1;
  wire _abc_896_n75;
  wire _abc_896_n76;
  wire _abc_896_n77;
  wire _abc_896_n78_1;
  wire _abc_896_n79;
  wire _abc_896_n80;
  wire _abc_896_n81_1;
  wire _abc_896_n82;
  wire _abc_896_n83;
  wire _abc_896_n84_1;
  wire _abc_896_n85;
  wire _abc_896_n86;
  wire _abc_896_n87_1;
  wire _abc_896_n88;
  wire _abc_896_n89;
  wire _abc_896_n90_1;
  wire _abc_896_n91;
  wire _abc_896_n92;
  wire _abc_896_n93_1;
  wire _abc_896_n94;
  wire _abc_896_n95;
  wire _abc_896_n96_1;
  wire _abc_896_n97;
  wire _abc_896_n98;
  wire _abc_896_n99_1;
  input clock;
  wire clock_bF_buf0;
  wire clock_bF_buf1;
  wire clock_bF_buf2;
  wire clock_bF_buf3;
  wire clock_bF_buf4;
  wire n10;
  wire n104;
  wire n109;
  wire n114;
  wire n119;
  wire n124;
  wire n129;
  wire n134;
  wire n139;
  wire n144;
  wire n15;
  wire n20;
  wire n25;
  wire n30;
  wire n35;
  wire n40;
  wire n45;
  wire n50;
  wire n55;
  wire n60;
  wire n65;
  wire n70;
  wire n75;
  wire n80;
  wire n85;
  wire n90;
  wire n95;
  wire n99;
  input nRESET_G;
  AND2X2 AND2X2_1 ( .A(_abc_896_n124), .B(OLD_REG_1_), .Y(_abc_896_n162) );
  AND2X2 AND2X2_10 ( .A(_abc_896_n111), .B(D_IN_REG_6_), .Y(_abc_896_n179) );
  AND2X2 AND2X2_11 ( .A(_abc_896_n124), .B(OLD_REG_6_), .Y(_abc_896_n182) );
  AND2X2 AND2X2_12 ( .A(_abc_896_n111), .B(D_IN_REG_7_), .Y(_abc_896_n183) );
  AND2X2 AND2X2_13 ( .A(_abc_896_n124), .B(OLD_REG_7_), .Y(_abc_896_n186) );
  AND2X2 AND2X2_14 ( .A(_abc_896_n111), .B(D_IN_REG_8_), .Y(_abc_896_n187) );
  AND2X2 AND2X2_15 ( .A(_abc_896_n114), .B(STATO_REG_0_), .Y(_abc_896_n190) );
  AND2X2 AND2X2_16 ( .A(STATO_REG_1_), .B(D_IN_REG_0_), .Y(_abc_896_n191) );
  AND2X2 AND2X2_17 ( .A(_abc_896_n150_1), .B(_abc_896_n192), .Y(_abc_896_n193) );
  AND2X2 AND2X2_18 ( .A(_abc_896_n193), .B(D_OUT_REG_0_), .Y(_abc_896_n194) );
  AND2X2 AND2X2_19 ( .A(_abc_896_n143_1), .B(_abc_896_n159), .Y(_abc_896_n195) );
  AND2X2 AND2X2_2 ( .A(_abc_896_n111), .B(D_IN_REG_2_), .Y(_abc_896_n163) );
  AND2X2 AND2X2_20 ( .A(_abc_896_n116), .B(D_OUT_REG_1_), .Y(_abc_896_n196) );
  AND2X2 AND2X2_21 ( .A(_abc_896_n193), .B(D_OUT_REG_1_), .Y(_abc_896_n200) );
  AND2X2 AND2X2_22 ( .A(_abc_896_n143_1), .B(_abc_896_n163), .Y(_abc_896_n201) );
  AND2X2 AND2X2_23 ( .A(_abc_896_n116), .B(D_OUT_REG_2_), .Y(_abc_896_n202) );
  AND2X2 AND2X2_24 ( .A(_abc_896_n193), .B(D_OUT_REG_2_), .Y(_abc_896_n206) );
  AND2X2 AND2X2_25 ( .A(_abc_896_n143_1), .B(_abc_896_n167), .Y(_abc_896_n207) );
  AND2X2 AND2X2_26 ( .A(_abc_896_n116), .B(D_OUT_REG_3_), .Y(_abc_896_n208) );
  AND2X2 AND2X2_27 ( .A(_abc_896_n193), .B(D_OUT_REG_3_), .Y(_abc_896_n212) );
  AND2X2 AND2X2_28 ( .A(_abc_896_n143_1), .B(_abc_896_n171), .Y(_abc_896_n213) );
  AND2X2 AND2X2_29 ( .A(_abc_896_n116), .B(D_OUT_REG_4_), .Y(_abc_896_n214) );
  AND2X2 AND2X2_3 ( .A(_abc_896_n124), .B(OLD_REG_2_), .Y(_abc_896_n166) );
  AND2X2 AND2X2_30 ( .A(_abc_896_n193), .B(D_OUT_REG_4_), .Y(_abc_896_n218) );
  AND2X2 AND2X2_31 ( .A(_abc_896_n143_1), .B(_abc_896_n175), .Y(_abc_896_n219) );
  AND2X2 AND2X2_32 ( .A(_abc_896_n116), .B(D_OUT_REG_5_), .Y(_abc_896_n220) );
  AND2X2 AND2X2_33 ( .A(_abc_896_n193), .B(D_OUT_REG_5_), .Y(_abc_896_n224) );
  AND2X2 AND2X2_34 ( .A(_abc_896_n143_1), .B(_abc_896_n179), .Y(_abc_896_n225) );
  AND2X2 AND2X2_35 ( .A(_abc_896_n116), .B(D_OUT_REG_6_), .Y(_abc_896_n226) );
  AND2X2 AND2X2_36 ( .A(_abc_896_n193), .B(D_OUT_REG_6_), .Y(_abc_896_n230) );
  AND2X2 AND2X2_37 ( .A(_abc_896_n143_1), .B(_abc_896_n183), .Y(_abc_896_n231) );
  AND2X2 AND2X2_38 ( .A(_abc_896_n116), .B(D_OUT_REG_7_), .Y(_abc_896_n232) );
  AND2X2 AND2X2_39 ( .A(_abc_896_n193), .B(D_OUT_REG_7_), .Y(_abc_896_n236) );
  AND2X2 AND2X2_4 ( .A(_abc_896_n111), .B(D_IN_REG_3_), .Y(_abc_896_n167) );
  AND2X2 AND2X2_40 ( .A(_abc_896_n143_1), .B(_abc_896_n187), .Y(_abc_896_n237) );
  AND2X2 AND2X2_41 ( .A(_abc_896_n59), .B(OLD_REG_4_), .Y(_abc_896_n60) );
  AND2X2 AND2X2_42 ( .A(_abc_896_n62), .B(D_IN_REG_5_), .Y(_abc_896_n63) );
  AND2X2 AND2X2_43 ( .A(_abc_896_n61_1), .B(_abc_896_n64_1), .Y(_abc_896_n65) );
  AND2X2 AND2X2_44 ( .A(OLD_REG_5_), .B(D_IN_REG_6_), .Y(_abc_896_n67_1) );
  AND2X2 AND2X2_45 ( .A(_abc_896_n68), .B(_abc_896_n66), .Y(_abc_896_n69) );
  AND2X2 AND2X2_46 ( .A(_abc_896_n65), .B(_abc_896_n70_1), .Y(_abc_896_n71) );
  AND2X2 AND2X2_47 ( .A(OLD_REG_0_), .B(D_IN_REG_1_), .Y(_abc_896_n73) );
  AND2X2 AND2X2_48 ( .A(_abc_896_n74_1), .B(_abc_896_n72), .Y(_abc_896_n75) );
  AND2X2 AND2X2_49 ( .A(OLD_REG_1_), .B(D_IN_REG_2_), .Y(_abc_896_n78_1) );
  AND2X2 AND2X2_5 ( .A(_abc_896_n124), .B(OLD_REG_3_), .Y(_abc_896_n170) );
  AND2X2 AND2X2_50 ( .A(_abc_896_n79), .B(_abc_896_n77), .Y(_abc_896_n80) );
  AND2X2 AND2X2_51 ( .A(_abc_896_n76), .B(_abc_896_n81_1), .Y(_abc_896_n82) );
  AND2X2 AND2X2_52 ( .A(_abc_896_n71), .B(_abc_896_n82), .Y(_abc_896_n83) );
  AND2X2 AND2X2_53 ( .A(OLD_REG_6_), .B(D_IN_REG_7_), .Y(_abc_896_n85) );
  AND2X2 AND2X2_54 ( .A(_abc_896_n86), .B(_abc_896_n84_1), .Y(_abc_896_n87_1) );
  AND2X2 AND2X2_55 ( .A(OLD_REG_7_), .B(D_IN_REG_8_), .Y(_abc_896_n90_1) );
  AND2X2 AND2X2_56 ( .A(_abc_896_n91), .B(_abc_896_n89), .Y(_abc_896_n92) );
  AND2X2 AND2X2_57 ( .A(_abc_896_n88), .B(_abc_896_n93_1), .Y(_abc_896_n94) );
  AND2X2 AND2X2_58 ( .A(OLD_REG_3_), .B(D_IN_REG_4_), .Y(_abc_896_n96_1) );
  AND2X2 AND2X2_59 ( .A(_abc_896_n97), .B(_abc_896_n95), .Y(_abc_896_n98) );
  AND2X2 AND2X2_6 ( .A(_abc_896_n111), .B(D_IN_REG_4_), .Y(_abc_896_n171) );
  AND2X2 AND2X2_60 ( .A(OLD_REG_2_), .B(D_IN_REG_3_), .Y(_abc_896_n101) );
  AND2X2 AND2X2_61 ( .A(_abc_896_n102), .B(_abc_896_n100), .Y(_abc_896_n103) );
  AND2X2 AND2X2_62 ( .A(_abc_896_n99_1), .B(_abc_896_n104), .Y(_abc_896_n105) );
  AND2X2 AND2X2_63 ( .A(_abc_896_n94), .B(_abc_896_n105), .Y(_abc_896_n106) );
  AND2X2 AND2X2_64 ( .A(_abc_896_n83), .B(_abc_896_n106), .Y(_abc_896_n107_1) );
  AND2X2 AND2X2_65 ( .A(STATO_REG_0_), .B(STATO_REG_1_), .Y(_abc_896_n108) );
  AND2X2 AND2X2_66 ( .A(_abc_896_n107_1), .B(_abc_896_n108), .Y(_abc_896_n109_1) );
  AND2X2 AND2X2_67 ( .A(STATO_REG_0_), .B(D_IN_REG_0_), .Y(_abc_896_n111) );
  AND2X2 AND2X2_68 ( .A(_abc_896_n114), .B(STATO_REG_1_), .Y(_abc_896_n115_1) );
  AND2X2 AND2X2_69 ( .A(_abc_896_n115_1), .B(_abc_896_n113_1), .Y(_abc_896_n116) );
  AND2X2 AND2X2_7 ( .A(_abc_896_n124), .B(OLD_REG_4_), .Y(_abc_896_n174) );
  AND2X2 AND2X2_70 ( .A(_abc_896_n117), .B(_abc_896_n112), .Y(_abc_896_n118) );
  AND2X2 AND2X2_71 ( .A(_abc_896_n112), .B(_abc_896_n123), .Y(_abc_896_n124) );
  AND2X2 AND2X2_72 ( .A(_abc_896_n124), .B(nRESET_G), .Y(_abc_896_n125_1) );
  AND2X2 AND2X2_73 ( .A(_abc_896_n125_1), .B(D_IN_REG_1_), .Y(n10) );
  AND2X2 AND2X2_74 ( .A(_abc_896_n125_1), .B(D_IN_REG_2_), .Y(n144) );
  AND2X2 AND2X2_75 ( .A(_abc_896_n125_1), .B(D_IN_REG_3_), .Y(n139) );
  AND2X2 AND2X2_76 ( .A(_abc_896_n125_1), .B(D_IN_REG_4_), .Y(n134) );
  AND2X2 AND2X2_77 ( .A(_abc_896_n125_1), .B(D_IN_REG_5_), .Y(n129) );
  AND2X2 AND2X2_78 ( .A(_abc_896_n125_1), .B(D_IN_REG_6_), .Y(n124) );
  AND2X2 AND2X2_79 ( .A(_abc_896_n125_1), .B(D_IN_REG_7_), .Y(n119) );
  AND2X2 AND2X2_8 ( .A(_abc_896_n111), .B(D_IN_REG_5_), .Y(_abc_896_n175) );
  AND2X2 AND2X2_80 ( .A(_abc_896_n125_1), .B(D_IN_REG_8_), .Y(n114) );
  AND2X2 AND2X2_81 ( .A(_abc_896_n143_1), .B(_abc_896_n111), .Y(_abc_896_n144) );
  AND2X2 AND2X2_82 ( .A(_abc_896_n134), .B(STATO_REG_0_), .Y(_abc_896_n145_1) );
  AND2X2 AND2X2_83 ( .A(_abc_896_n146), .B(X), .Y(_abc_896_n147) );
  AND2X2 AND2X2_84 ( .A(_abc_896_n116), .B(D_OUT_REG_0_), .Y(_abc_896_n152) );
  AND2X2 AND2X2_85 ( .A(_abc_896_n145_1), .B(_abc_896_n153), .Y(_abc_896_n154) );
  AND2X2 AND2X2_86 ( .A(_abc_896_n124), .B(OLD_REG_0_), .Y(_abc_896_n158) );
  AND2X2 AND2X2_87 ( .A(_abc_896_n111), .B(D_IN_REG_1_), .Y(_abc_896_n159) );
  AND2X2 AND2X2_9 ( .A(_abc_896_n124), .B(OLD_REG_5_), .Y(_abc_896_n178) );
  DFFPOSX1 DFFPOSX1_1 ( .CLK(clock), .D(n95), .Q(Y_REG) );
  DFFPOSX1 DFFPOSX1_10 ( .CLK(clock), .D(n50), .Q(D_OUT_REG_0_) );
  DFFPOSX1 DFFPOSX1_11 ( .CLK(clock), .D(n55), .Q(OLD_REG_7_) );
  DFFPOSX1 DFFPOSX1_12 ( .CLK(clock), .D(n60), .Q(OLD_REG_6_) );
  DFFPOSX1 DFFPOSX1_13 ( .CLK(clock), .D(n65), .Q(OLD_REG_5_) );
  DFFPOSX1 DFFPOSX1_14 ( .CLK(clock), .D(n70), .Q(OLD_REG_4_) );
  DFFPOSX1 DFFPOSX1_15 ( .CLK(clock), .D(n75), .Q(OLD_REG_3_) );
  DFFPOSX1 DFFPOSX1_16 ( .CLK(clock), .D(n80), .Q(OLD_REG_2_) );
  DFFPOSX1 DFFPOSX1_17 ( .CLK(clock), .D(n85), .Q(OLD_REG_1_) );
  DFFPOSX1 DFFPOSX1_18 ( .CLK(clock), .D(n90), .Q(OLD_REG_0_) );
  DFFPOSX1 DFFPOSX1_19 ( .CLK(clock), .D(n99), .Q(STATO_REG_1_) );
  DFFPOSX1 DFFPOSX1_2 ( .CLK(clock), .D(n10), .Q(D_IN_REG_0_) );
  DFFPOSX1 DFFPOSX1_20 ( .CLK(clock), .D(n104), .Q(STATO_REG_0_) );
  DFFPOSX1 DFFPOSX1_21 ( .CLK(clock), .D(n109), .Q(D_IN_REG_8_) );
  DFFPOSX1 DFFPOSX1_22 ( .CLK(clock), .D(n114), .Q(D_IN_REG_7_) );
  DFFPOSX1 DFFPOSX1_23 ( .CLK(clock), .D(n119), .Q(D_IN_REG_6_) );
  DFFPOSX1 DFFPOSX1_24 ( .CLK(clock), .D(n124), .Q(D_IN_REG_5_) );
  DFFPOSX1 DFFPOSX1_25 ( .CLK(clock), .D(n129), .Q(D_IN_REG_4_) );
  DFFPOSX1 DFFPOSX1_26 ( .CLK(clock), .D(n134), .Q(D_IN_REG_3_) );
  DFFPOSX1 DFFPOSX1_27 ( .CLK(clock), .D(n139), .Q(D_IN_REG_2_) );
  DFFPOSX1 DFFPOSX1_28 ( .CLK(clock), .D(n144), .Q(D_IN_REG_1_) );
  DFFPOSX1 DFFPOSX1_3 ( .CLK(clock), .D(n15), .Q(D_OUT_REG_7_) );
  DFFPOSX1 DFFPOSX1_4 ( .CLK(clock), .D(n20), .Q(D_OUT_REG_6_) );
  DFFPOSX1 DFFPOSX1_5 ( .CLK(clock), .D(n25), .Q(D_OUT_REG_5_) );
  DFFPOSX1 DFFPOSX1_6 ( .CLK(clock), .D(n30), .Q(D_OUT_REG_4_) );
  DFFPOSX1 DFFPOSX1_7 ( .CLK(clock), .D(n35), .Q(D_OUT_REG_3_) );
  DFFPOSX1 DFFPOSX1_8 ( .CLK(clock), .D(n40), .Q(D_OUT_REG_2_) );
  DFFPOSX1 DFFPOSX1_9 ( .CLK(clock), .D(n45), .Q(D_OUT_REG_1_) );
  INVX1 INVX1_1 ( .A(D_IN_REG_5_), .Y(_abc_896_n59) );
  INVX1 INVX1_10 ( .A(_abc_896_n80), .Y(_abc_896_n81_1) );
  INVX1 INVX1_11 ( .A(_abc_896_n85), .Y(_abc_896_n86) );
  INVX1 INVX1_12 ( .A(_abc_896_n87_1), .Y(_abc_896_n88) );
  INVX1 INVX1_13 ( .A(_abc_896_n90_1), .Y(_abc_896_n91) );
  INVX1 INVX1_14 ( .A(_abc_896_n92), .Y(_abc_896_n93_1) );
  INVX1 INVX1_15 ( .A(_abc_896_n96_1), .Y(_abc_896_n97) );
  INVX1 INVX1_16 ( .A(_abc_896_n98), .Y(_abc_896_n99_1) );
  INVX1 INVX1_17 ( .A(_abc_896_n101), .Y(_abc_896_n102) );
  INVX1 INVX1_18 ( .A(_abc_896_n103), .Y(_abc_896_n104) );
  INVX1 INVX1_19 ( .A(_abc_896_n111), .Y(_abc_896_n112) );
  INVX1 INVX1_2 ( .A(_abc_896_n60), .Y(_abc_896_n61_1) );
  INVX1 INVX1_20 ( .A(STATO_REG_0_), .Y(_abc_896_n113_1) );
  INVX1 INVX1_21 ( .A(D_IN_REG_0_), .Y(_abc_896_n114) );
  INVX1 INVX1_22 ( .A(_abc_896_n116), .Y(_abc_896_n117) );
  INVX1 INVX1_23 ( .A(STATO_REG_1_), .Y(_abc_896_n134) );
  INVX1 INVX1_24 ( .A(_abc_896_n150_1), .Y(_abc_896_n151) );
  INVX1 INVX1_3 ( .A(OLD_REG_4_), .Y(_abc_896_n62) );
  INVX1 INVX1_4 ( .A(_abc_896_n63), .Y(_abc_896_n64_1) );
  INVX1 INVX1_5 ( .A(_abc_896_n67_1), .Y(_abc_896_n68) );
  INVX1 INVX1_6 ( .A(_abc_896_n69), .Y(_abc_896_n70_1) );
  INVX1 INVX1_7 ( .A(_abc_896_n73), .Y(_abc_896_n74_1) );
  INVX1 INVX1_8 ( .A(_abc_896_n75), .Y(_abc_896_n76) );
  INVX1 INVX1_9 ( .A(_abc_896_n78_1), .Y(_abc_896_n79) );
  INVX8 INVX8_1 ( .A(nRESET_G), .Y(_abc_896_n110) );
  OR2X2 OR2X2_1 ( .A(_abc_896_n163), .B(_abc_896_n110), .Y(_abc_896_n164) );
  OR2X2 OR2X2_10 ( .A(_abc_896_n178), .B(_abc_896_n180), .Y(n65) );
  OR2X2 OR2X2_11 ( .A(_abc_896_n183), .B(_abc_896_n110), .Y(_abc_896_n184) );
  OR2X2 OR2X2_12 ( .A(_abc_896_n182), .B(_abc_896_n184), .Y(n60) );
  OR2X2 OR2X2_13 ( .A(_abc_896_n187), .B(_abc_896_n110), .Y(_abc_896_n188) );
  OR2X2 OR2X2_14 ( .A(_abc_896_n186), .B(_abc_896_n188), .Y(n55) );
  OR2X2 OR2X2_15 ( .A(_abc_896_n190), .B(_abc_896_n191), .Y(_abc_896_n192) );
  OR2X2 OR2X2_16 ( .A(_abc_896_n196), .B(_abc_896_n110), .Y(_abc_896_n197) );
  OR2X2 OR2X2_17 ( .A(_abc_896_n195), .B(_abc_896_n197), .Y(_abc_896_n198) );
  OR2X2 OR2X2_18 ( .A(_abc_896_n198), .B(_abc_896_n194), .Y(n50) );
  OR2X2 OR2X2_19 ( .A(_abc_896_n202), .B(_abc_896_n110), .Y(_abc_896_n203) );
  OR2X2 OR2X2_2 ( .A(_abc_896_n162), .B(_abc_896_n164), .Y(n85) );
  OR2X2 OR2X2_20 ( .A(_abc_896_n201), .B(_abc_896_n203), .Y(_abc_896_n204) );
  OR2X2 OR2X2_21 ( .A(_abc_896_n204), .B(_abc_896_n200), .Y(n45) );
  OR2X2 OR2X2_22 ( .A(_abc_896_n208), .B(_abc_896_n110), .Y(_abc_896_n209) );
  OR2X2 OR2X2_23 ( .A(_abc_896_n207), .B(_abc_896_n209), .Y(_abc_896_n210) );
  OR2X2 OR2X2_24 ( .A(_abc_896_n210), .B(_abc_896_n206), .Y(n40) );
  OR2X2 OR2X2_25 ( .A(_abc_896_n214), .B(_abc_896_n110), .Y(_abc_896_n215) );
  OR2X2 OR2X2_26 ( .A(_abc_896_n213), .B(_abc_896_n215), .Y(_abc_896_n216) );
  OR2X2 OR2X2_27 ( .A(_abc_896_n216), .B(_abc_896_n212), .Y(n35) );
  OR2X2 OR2X2_28 ( .A(_abc_896_n220), .B(_abc_896_n110), .Y(_abc_896_n221) );
  OR2X2 OR2X2_29 ( .A(_abc_896_n219), .B(_abc_896_n221), .Y(_abc_896_n222) );
  OR2X2 OR2X2_3 ( .A(_abc_896_n167), .B(_abc_896_n110), .Y(_abc_896_n168) );
  OR2X2 OR2X2_30 ( .A(_abc_896_n222), .B(_abc_896_n218), .Y(n30) );
  OR2X2 OR2X2_31 ( .A(_abc_896_n226), .B(_abc_896_n110), .Y(_abc_896_n227) );
  OR2X2 OR2X2_32 ( .A(_abc_896_n225), .B(_abc_896_n227), .Y(_abc_896_n228) );
  OR2X2 OR2X2_33 ( .A(_abc_896_n228), .B(_abc_896_n224), .Y(n25) );
  OR2X2 OR2X2_34 ( .A(_abc_896_n232), .B(_abc_896_n110), .Y(_abc_896_n233) );
  OR2X2 OR2X2_35 ( .A(_abc_896_n231), .B(_abc_896_n233), .Y(_abc_896_n234) );
  OR2X2 OR2X2_36 ( .A(_abc_896_n234), .B(_abc_896_n230), .Y(n20) );
  OR2X2 OR2X2_37 ( .A(_abc_896_n237), .B(_abc_896_n110), .Y(_abc_896_n238) );
  OR2X2 OR2X2_38 ( .A(_abc_896_n238), .B(_abc_896_n236), .Y(n15) );
  OR2X2 OR2X2_39 ( .A(OLD_REG_5_), .B(D_IN_REG_6_), .Y(_abc_896_n66) );
  OR2X2 OR2X2_4 ( .A(_abc_896_n166), .B(_abc_896_n168), .Y(n80) );
  OR2X2 OR2X2_40 ( .A(OLD_REG_0_), .B(D_IN_REG_1_), .Y(_abc_896_n72) );
  OR2X2 OR2X2_41 ( .A(OLD_REG_1_), .B(D_IN_REG_2_), .Y(_abc_896_n77) );
  OR2X2 OR2X2_42 ( .A(OLD_REG_6_), .B(D_IN_REG_7_), .Y(_abc_896_n84_1) );
  OR2X2 OR2X2_43 ( .A(OLD_REG_7_), .B(D_IN_REG_8_), .Y(_abc_896_n89) );
  OR2X2 OR2X2_44 ( .A(OLD_REG_3_), .B(D_IN_REG_4_), .Y(_abc_896_n95) );
  OR2X2 OR2X2_45 ( .A(OLD_REG_2_), .B(D_IN_REG_3_), .Y(_abc_896_n100) );
  OR2X2 OR2X2_46 ( .A(_abc_896_n118), .B(_abc_896_n110), .Y(_abc_896_n119_1) );
  OR2X2 OR2X2_47 ( .A(_abc_896_n109_1), .B(_abc_896_n119_1), .Y(n104) );
  OR2X2 OR2X2_48 ( .A(_abc_896_n110), .B(STATO_REG_1_), .Y(_abc_896_n121_1) );
  OR2X2 OR2X2_49 ( .A(_abc_896_n121_1), .B(_abc_896_n111), .Y(n99) );
  OR2X2 OR2X2_5 ( .A(_abc_896_n171), .B(_abc_896_n110), .Y(_abc_896_n172) );
  OR2X2 OR2X2_50 ( .A(STATO_REG_0_), .B(STATO_REG_1_), .Y(_abc_896_n123) );
  OR2X2 OR2X2_51 ( .A(_abc_896_n60), .B(_abc_896_n63), .Y(_abc_896_n135) );
  OR2X2 OR2X2_52 ( .A(_abc_896_n135), .B(_abc_896_n69), .Y(_abc_896_n136) );
  OR2X2 OR2X2_53 ( .A(_abc_896_n75), .B(_abc_896_n80), .Y(_abc_896_n137_1) );
  OR2X2 OR2X2_54 ( .A(_abc_896_n136), .B(_abc_896_n137_1), .Y(_abc_896_n138) );
  OR2X2 OR2X2_55 ( .A(_abc_896_n87_1), .B(_abc_896_n92), .Y(_abc_896_n139_1) );
  OR2X2 OR2X2_56 ( .A(_abc_896_n98), .B(_abc_896_n103), .Y(_abc_896_n140) );
  OR2X2 OR2X2_57 ( .A(_abc_896_n139_1), .B(_abc_896_n140), .Y(_abc_896_n141) );
  OR2X2 OR2X2_58 ( .A(_abc_896_n138), .B(_abc_896_n141), .Y(_abc_896_n142) );
  OR2X2 OR2X2_59 ( .A(_abc_896_n142), .B(_abc_896_n134), .Y(_abc_896_n143_1) );
  OR2X2 OR2X2_6 ( .A(_abc_896_n170), .B(_abc_896_n172), .Y(n75) );
  OR2X2 OR2X2_60 ( .A(_abc_896_n124), .B(_abc_896_n145_1), .Y(_abc_896_n146) );
  OR2X2 OR2X2_61 ( .A(_abc_896_n147), .B(_abc_896_n110), .Y(_abc_896_n148) );
  OR2X2 OR2X2_62 ( .A(_abc_896_n144), .B(_abc_896_n148), .Y(n109) );
  OR2X2 OR2X2_63 ( .A(_abc_896_n107_1), .B(_abc_896_n112), .Y(_abc_896_n150_1) );
  OR2X2 OR2X2_64 ( .A(D_IN_REG_0_), .B(Y_REG), .Y(_abc_896_n153) );
  OR2X2 OR2X2_65 ( .A(_abc_896_n154), .B(_abc_896_n110), .Y(_abc_896_n155) );
  OR2X2 OR2X2_66 ( .A(_abc_896_n155), .B(_abc_896_n152), .Y(_abc_896_n156) );
  OR2X2 OR2X2_67 ( .A(_abc_896_n151), .B(_abc_896_n156), .Y(n95) );
  OR2X2 OR2X2_68 ( .A(_abc_896_n159), .B(_abc_896_n110), .Y(_abc_896_n160) );
  OR2X2 OR2X2_69 ( .A(_abc_896_n158), .B(_abc_896_n160), .Y(n90) );
  OR2X2 OR2X2_7 ( .A(_abc_896_n175), .B(_abc_896_n110), .Y(_abc_896_n176) );
  OR2X2 OR2X2_8 ( .A(_abc_896_n174), .B(_abc_896_n176), .Y(n70) );
  OR2X2 OR2X2_9 ( .A(_abc_896_n179), .B(_abc_896_n110), .Y(_abc_896_n180) );
endmodule