module b01_reset(clock, RESET_G, nRESET_G, LINE1, LINE2, OUTP_REG, OVERFLW_REG);

input LINE1;
input LINE2;
output OUTP_REG;
output OVERFLW_REG;
input RESET_G;
wire STATO_REG_0_; 
wire STATO_REG_1_; 
wire STATO_REG_2_; 
wire _abc_289_new_n12_; 
wire _abc_289_new_n13_; 
wire _abc_289_new_n14_; 
wire _abc_289_new_n16_; 
wire _abc_289_new_n17_; 
wire _abc_289_new_n18_; 
wire _abc_289_new_n19_; 
wire _abc_289_new_n20_; 
wire _abc_289_new_n21_; 
wire _abc_289_new_n22_; 
wire _abc_289_new_n23_; 
wire _abc_289_new_n24_; 
wire _abc_289_new_n25_; 
wire _abc_289_new_n26_; 
wire _abc_289_new_n27_; 
wire _abc_289_new_n28_; 
wire _abc_289_new_n29_; 
wire _abc_289_new_n30_; 
wire _abc_289_new_n31_; 
wire _abc_289_new_n32_; 
wire _abc_289_new_n33_; 
wire _abc_289_new_n34_; 
wire _abc_289_new_n35_; 
wire _abc_289_new_n36_; 
wire _abc_289_new_n38_; 
wire _abc_289_new_n39_; 
wire _abc_289_new_n40_; 
wire _abc_289_new_n41_; 
wire _abc_289_new_n42_; 
wire _abc_289_new_n43_; 
wire _abc_289_new_n44_; 
wire _abc_289_new_n45_; 
wire _abc_289_new_n46_; 
wire _abc_289_new_n47_; 
wire _abc_289_new_n48_; 
wire _abc_289_new_n50_; 
wire _abc_289_new_n51_; 
wire _abc_289_new_n52_; 
wire _abc_289_new_n53_; 
wire _abc_289_new_n54_; 
wire _abc_289_new_n56_; 
wire _abc_289_new_n57_; 
wire _abc_289_new_n58_; 
wire _abc_289_new_n59_; 
wire _abc_289_new_n60_; 
wire _abc_289_new_n61_; 
wire _auto_iopadmap_cc_368_execute_341; 
wire _auto_iopadmap_cc_368_execute_343; 
input clock;
wire n14; 
wire n18; 
wire n23; 
wire n28; 
wire n33; 
input nRESET_G;
AND2X2 AND2X2_1 ( .A(_abc_289_new_n12_), .B(STATO_REG_0_), .Y(_abc_289_new_n13_));
AND2X2 AND2X2_10 ( .A(_abc_289_new_n30_), .B(_abc_289_new_n26_), .Y(_abc_289_new_n31_));
AND2X2 AND2X2_11 ( .A(n14), .B(_abc_289_new_n32_), .Y(_abc_289_new_n33_));
AND2X2 AND2X2_12 ( .A(_abc_289_new_n19_), .B(_abc_289_new_n24_), .Y(_abc_289_new_n34_));
AND2X2 AND2X2_13 ( .A(_abc_289_new_n38_), .B(_abc_289_new_n28_), .Y(_abc_289_new_n39_));
AND2X2 AND2X2_14 ( .A(_abc_289_new_n17_), .B(STATO_REG_0_), .Y(_abc_289_new_n40_));
AND2X2 AND2X2_15 ( .A(_abc_289_new_n41_), .B(STATO_REG_2_), .Y(_abc_289_new_n42_));
AND2X2 AND2X2_16 ( .A(_abc_289_new_n32_), .B(STATO_REG_0_), .Y(_abc_289_new_n43_));
AND2X2 AND2X2_17 ( .A(_abc_289_new_n43_), .B(_abc_289_new_n28_), .Y(_abc_289_new_n44_));
AND2X2 AND2X2_18 ( .A(_abc_289_new_n45_), .B(_abc_289_new_n19_), .Y(_abc_289_new_n46_));
AND2X2 AND2X2_19 ( .A(_abc_289_new_n32_), .B(_abc_289_new_n17_), .Y(_abc_289_new_n50_));
AND2X2 AND2X2_2 ( .A(nRESET_G), .B(STATO_REG_1_), .Y(_abc_289_new_n14_));
AND2X2 AND2X2_20 ( .A(_abc_289_new_n20_), .B(_abc_289_new_n50_), .Y(_abc_289_new_n51_));
AND2X2 AND2X2_21 ( .A(_abc_289_new_n21_), .B(_abc_289_new_n52_), .Y(_abc_289_new_n53_));
AND2X2 AND2X2_22 ( .A(_abc_289_new_n56_), .B(_abc_289_new_n12_), .Y(_abc_289_new_n57_));
AND2X2 AND2X2_23 ( .A(_abc_289_new_n28_), .B(STATO_REG_2_), .Y(_abc_289_new_n60_));
AND2X2 AND2X2_24 ( .A(_abc_289_new_n59_), .B(_abc_289_new_n60_), .Y(_abc_289_new_n61_));
AND2X2 AND2X2_3 ( .A(_abc_289_new_n13_), .B(_abc_289_new_n14_), .Y(n14));
AND2X2 AND2X2_4 ( .A(_abc_289_new_n18_), .B(STATO_REG_1_), .Y(_abc_289_new_n19_));
AND2X2 AND2X2_5 ( .A(_abc_289_new_n21_), .B(_abc_289_new_n17_), .Y(_abc_289_new_n22_));
AND2X2 AND2X2_6 ( .A(LINE2), .B(LINE1), .Y(_abc_289_new_n24_));
AND2X2 AND2X2_7 ( .A(_abc_289_new_n24_), .B(_abc_289_new_n18_), .Y(_abc_289_new_n25_));
AND2X2 AND2X2_8 ( .A(_abc_289_new_n12_), .B(_abc_289_new_n28_), .Y(_abc_289_new_n29_));
AND2X2 AND2X2_9 ( .A(_abc_289_new_n27_), .B(_abc_289_new_n29_), .Y(_abc_289_new_n30_));
BUFX2 BUFX2_1 ( .A(_auto_iopadmap_cc_368_execute_341), .Y(OUTP_REG));
BUFX2 BUFX2_2 ( .A(_auto_iopadmap_cc_368_execute_343), .Y(OVERFLW_REG));
DFFPOSX1 DFFPOSX1_1 ( .CLK(clock), .D(n18), .Q(STATO_REG_2_));
DFFPOSX1 DFFPOSX1_2 ( .CLK(clock), .D(n14), .Q(_auto_iopadmap_cc_368_execute_343));
DFFPOSX1 DFFPOSX1_3 ( .CLK(clock), .D(n28), .Q(STATO_REG_0_));
DFFPOSX1 DFFPOSX1_4 ( .CLK(clock), .D(n33), .Q(_auto_iopadmap_cc_368_execute_341));
DFFPOSX1 DFFPOSX1_5 ( .CLK(clock), .D(n23), .Q(STATO_REG_1_));
INVX1 INVX1_1 ( .A(STATO_REG_0_), .Y(_abc_289_new_n18_));
INVX1 INVX1_2 ( .A(_abc_289_new_n20_), .Y(_abc_289_new_n21_));
INVX1 INVX1_3 ( .A(_abc_289_new_n25_), .Y(_abc_289_new_n26_));
INVX1 INVX1_4 ( .A(STATO_REG_1_), .Y(_abc_289_new_n28_));
INVX1 INVX1_5 ( .A(_abc_289_new_n24_), .Y(_abc_289_new_n32_));
INVX1 INVX1_6 ( .A(_abc_289_new_n17_), .Y(_abc_289_new_n38_));
INVX1 INVX1_7 ( .A(_abc_289_new_n50_), .Y(_abc_289_new_n52_));
INVX2 INVX2_1 ( .A(STATO_REG_2_), .Y(_abc_289_new_n12_));
INVX2 INVX2_2 ( .A(nRESET_G), .Y(_abc_289_new_n16_));
OR2X2 OR2X2_1 ( .A(LINE2), .B(LINE1), .Y(_abc_289_new_n17_));
OR2X2 OR2X2_10 ( .A(_abc_289_new_n44_), .B(_abc_289_new_n46_), .Y(_abc_289_new_n47_));
OR2X2 OR2X2_11 ( .A(_abc_289_new_n47_), .B(_abc_289_new_n16_), .Y(_abc_289_new_n48_));
OR2X2 OR2X2_12 ( .A(_abc_289_new_n48_), .B(_abc_289_new_n42_), .Y(n23));
OR2X2 OR2X2_13 ( .A(_abc_289_new_n53_), .B(_abc_289_new_n16_), .Y(_abc_289_new_n54_));
OR2X2 OR2X2_14 ( .A(_abc_289_new_n54_), .B(_abc_289_new_n51_), .Y(n33));
OR2X2 OR2X2_15 ( .A(_abc_289_new_n19_), .B(_abc_289_new_n24_), .Y(_abc_289_new_n56_));
OR2X2 OR2X2_16 ( .A(_abc_289_new_n57_), .B(_abc_289_new_n16_), .Y(_abc_289_new_n58_));
OR2X2 OR2X2_17 ( .A(_abc_289_new_n17_), .B(STATO_REG_0_), .Y(_abc_289_new_n59_));
OR2X2 OR2X2_18 ( .A(_abc_289_new_n58_), .B(_abc_289_new_n61_), .Y(n18));
OR2X2 OR2X2_2 ( .A(_abc_289_new_n19_), .B(_abc_289_new_n12_), .Y(_abc_289_new_n20_));
OR2X2 OR2X2_3 ( .A(_abc_289_new_n22_), .B(_abc_289_new_n16_), .Y(_abc_289_new_n23_));
OR2X2 OR2X2_4 ( .A(_abc_289_new_n24_), .B(_abc_289_new_n18_), .Y(_abc_289_new_n27_));
OR2X2 OR2X2_5 ( .A(_abc_289_new_n33_), .B(_abc_289_new_n34_), .Y(_abc_289_new_n35_));
OR2X2 OR2X2_6 ( .A(_abc_289_new_n35_), .B(_abc_289_new_n31_), .Y(_abc_289_new_n36_));
OR2X2 OR2X2_7 ( .A(_abc_289_new_n36_), .B(_abc_289_new_n23_), .Y(n28));
OR2X2 OR2X2_8 ( .A(_abc_289_new_n39_), .B(_abc_289_new_n40_), .Y(_abc_289_new_n41_));
OR2X2 OR2X2_9 ( .A(_abc_289_new_n24_), .B(_abc_289_new_n12_), .Y(_abc_289_new_n45_));


endmodule