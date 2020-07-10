module b06_reset(clock, RESET_G, nRESET_G, EQL, CONT_EQL, CC_MUX_REG_2_, CC_MUX_REG_1_, USCITE_REG_2_, USCITE_REG_1_, ENABLE_COUNT_REG, ACKOUT_REG);

output ACKOUT_REG;
output CC_MUX_REG_1_;
output CC_MUX_REG_2_;
input CONT_EQL;
output ENABLE_COUNT_REG;
input EQL;
input RESET_G;
wire STATE_REG_0_; 
wire STATE_REG_1_; 
wire STATE_REG_2_; 
output USCITE_REG_1_;
output USCITE_REG_2_;
wire _abc_317_new_n15_; 
wire _abc_317_new_n16_; 
wire _abc_317_new_n17_; 
wire _abc_317_new_n18_; 
wire _abc_317_new_n19_; 
wire _abc_317_new_n20_; 
wire _abc_317_new_n21_; 
wire _abc_317_new_n22_; 
wire _abc_317_new_n23_; 
wire _abc_317_new_n24_; 
wire _abc_317_new_n25_; 
wire _abc_317_new_n27_; 
wire _abc_317_new_n28_; 
wire _abc_317_new_n29_; 
wire _abc_317_new_n30_; 
wire _abc_317_new_n31_; 
wire _abc_317_new_n32_; 
wire _abc_317_new_n33_; 
wire _abc_317_new_n34_; 
wire _abc_317_new_n35_; 
wire _abc_317_new_n37_; 
wire _abc_317_new_n38_; 
wire _abc_317_new_n39_; 
wire _abc_317_new_n40_; 
wire _abc_317_new_n41_; 
wire _abc_317_new_n42_; 
wire _abc_317_new_n43_; 
wire _abc_317_new_n44_; 
wire _abc_317_new_n45_; 
wire _abc_317_new_n47_; 
wire _abc_317_new_n48_; 
wire _abc_317_new_n49_; 
wire _abc_317_new_n50_; 
wire _abc_317_new_n51_; 
wire _abc_317_new_n52_; 
wire _abc_317_new_n53_; 
wire _abc_317_new_n55_; 
wire _abc_317_new_n56_; 
wire _abc_317_new_n57_; 
wire _abc_317_new_n58_; 
wire _abc_317_new_n59_; 
wire _abc_317_new_n61_; 
wire _abc_317_new_n62_; 
wire _abc_317_new_n64_; 
wire _abc_317_new_n65_; 
wire _abc_317_new_n66_; 
wire _abc_317_new_n68_; 
wire _abc_317_new_n69_; 
wire _abc_317_new_n70_; 
wire _abc_317_new_n71_; 
wire _abc_317_new_n72_; 
input clock;
wire n22; 
wire n26; 
wire n31; 
wire n36; 
wire n41; 
wire n45; 
wire n49; 
wire n53; 
input nRESET_G;
AND2X2 AND2X2_1 ( .A(_abc_317_new_n17_), .B(_abc_317_new_n19_), .Y(_abc_317_new_n20_));
AND2X2 AND2X2_10 ( .A(STATE_REG_0_), .B(STATE_REG_2_), .Y(_abc_317_new_n49_));
AND2X2 AND2X2_11 ( .A(_abc_317_new_n15_), .B(nRESET_G), .Y(_abc_317_new_n51_));
AND2X2 AND2X2_12 ( .A(_abc_317_new_n51_), .B(_abc_317_new_n50_), .Y(_abc_317_new_n52_));
AND2X2 AND2X2_13 ( .A(_abc_317_new_n52_), .B(_abc_317_new_n48_), .Y(_abc_317_new_n53_));
AND2X2 AND2X2_14 ( .A(_abc_317_new_n28_), .B(nRESET_G), .Y(_abc_317_new_n55_));
AND2X2 AND2X2_15 ( .A(_abc_317_new_n56_), .B(_abc_317_new_n57_), .Y(_abc_317_new_n58_));
AND2X2 AND2X2_16 ( .A(_abc_317_new_n55_), .B(_abc_317_new_n58_), .Y(_abc_317_new_n59_));
AND2X2 AND2X2_17 ( .A(_abc_317_new_n39_), .B(STATE_REG_2_), .Y(_abc_317_new_n61_));
AND2X2 AND2X2_18 ( .A(_abc_317_new_n17_), .B(_abc_317_new_n42_), .Y(_abc_317_new_n64_));
AND2X2 AND2X2_19 ( .A(EQL), .B(nRESET_G), .Y(_abc_317_new_n65_));
AND2X2 AND2X2_2 ( .A(_abc_317_new_n23_), .B(_abc_317_new_n22_), .Y(_abc_317_new_n24_));
AND2X2 AND2X2_20 ( .A(_abc_317_new_n49_), .B(STATE_REG_1_), .Y(_abc_317_new_n69_));
AND2X2 AND2X2_21 ( .A(_abc_317_new_n70_), .B(nRESET_G), .Y(_abc_317_new_n71_));
AND2X2 AND2X2_22 ( .A(_abc_317_new_n71_), .B(_abc_317_new_n68_), .Y(_abc_317_new_n72_));
AND2X2 AND2X2_3 ( .A(_abc_317_new_n18_), .B(_abc_317_new_n22_), .Y(_abc_317_new_n33_));
AND2X2 AND2X2_4 ( .A(_abc_317_new_n32_), .B(_abc_317_new_n33_), .Y(_abc_317_new_n34_));
AND2X2 AND2X2_5 ( .A(_abc_317_new_n30_), .B(STATE_REG_2_), .Y(_abc_317_new_n37_));
AND2X2 AND2X2_6 ( .A(_abc_317_new_n33_), .B(_abc_317_new_n37_), .Y(_abc_317_new_n38_));
AND2X2 AND2X2_7 ( .A(STATE_REG_1_), .B(EQL), .Y(_abc_317_new_n39_));
AND2X2 AND2X2_8 ( .A(_abc_317_new_n32_), .B(EQL), .Y(_abc_317_new_n44_));
AND2X2 AND2X2_9 ( .A(_abc_317_new_n39_), .B(_abc_317_new_n30_), .Y(_abc_317_new_n47_));
DFFPOSX1 DFFPOSX1_1 ( .CLK(clock), .D(n49), .Q(USCITE_REG_2_));
DFFPOSX1 DFFPOSX1_2 ( .CLK(clock), .D(n41), .Q(CC_MUX_REG_2_));
DFFPOSX1 DFFPOSX1_3 ( .CLK(clock), .D(n45), .Q(CC_MUX_REG_1_));
DFFPOSX1 DFFPOSX1_4 ( .CLK(clock), .D(n22), .Q(ACKOUT_REG));
DFFPOSX1 DFFPOSX1_5 ( .CLK(clock), .D(n53), .Q(USCITE_REG_1_));
DFFPOSX1 DFFPOSX1_6 ( .CLK(clock), .D(n26), .Q(STATE_REG_2_));
DFFPOSX1 DFFPOSX1_7 ( .CLK(clock), .D(n31), .Q(STATE_REG_1_));
DFFPOSX1 DFFPOSX1_8 ( .CLK(clock), .D(n36), .Q(STATE_REG_0_));
INVX1 INVX1_1 ( .A(_abc_317_new_n15_), .Y(_abc_317_new_n16_));
INVX1 INVX1_10 ( .A(_abc_317_new_n49_), .Y(_abc_317_new_n50_));
INVX1 INVX1_11 ( .A(_abc_317_new_n53_), .Y(n45));
INVX1 INVX1_12 ( .A(_abc_317_new_n47_), .Y(_abc_317_new_n57_));
INVX1 INVX1_13 ( .A(_abc_317_new_n59_), .Y(n41));
INVX1 INVX1_14 ( .A(_abc_317_new_n65_), .Y(_abc_317_new_n66_));
INVX1 INVX1_15 ( .A(_abc_317_new_n72_), .Y(n22));
INVX1 INVX1_2 ( .A(STATE_REG_1_), .Y(_abc_317_new_n18_));
INVX1 INVX1_3 ( .A(nRESET_G), .Y(_abc_317_new_n21_));
INVX1 INVX1_4 ( .A(EQL), .Y(_abc_317_new_n22_));
INVX1 INVX1_5 ( .A(STATE_REG_2_), .Y(_abc_317_new_n27_));
INVX1 INVX1_6 ( .A(_abc_317_new_n28_), .Y(_abc_317_new_n29_));
INVX1 INVX1_7 ( .A(STATE_REG_0_), .Y(_abc_317_new_n30_));
INVX1 INVX1_8 ( .A(_abc_317_new_n31_), .Y(_abc_317_new_n32_));
INVX1 INVX1_9 ( .A(_abc_317_new_n42_), .Y(_abc_317_new_n43_));
OR2X2 OR2X2_1 ( .A(STATE_REG_0_), .B(STATE_REG_2_), .Y(_abc_317_new_n15_));
OR2X2 OR2X2_10 ( .A(_abc_317_new_n35_), .B(_abc_317_new_n29_), .Y(n26));
OR2X2 OR2X2_11 ( .A(_abc_317_new_n39_), .B(_abc_317_new_n21_), .Y(_abc_317_new_n40_));
OR2X2 OR2X2_12 ( .A(_abc_317_new_n38_), .B(_abc_317_new_n40_), .Y(_abc_317_new_n41_));
OR2X2 OR2X2_13 ( .A(_abc_317_new_n15_), .B(_abc_317_new_n18_), .Y(_abc_317_new_n42_));
OR2X2 OR2X2_14 ( .A(_abc_317_new_n44_), .B(_abc_317_new_n43_), .Y(_abc_317_new_n45_));
OR2X2 OR2X2_15 ( .A(_abc_317_new_n45_), .B(_abc_317_new_n41_), .Y(n31));
OR2X2 OR2X2_16 ( .A(_abc_317_new_n47_), .B(_abc_317_new_n33_), .Y(_abc_317_new_n48_));
OR2X2 OR2X2_17 ( .A(_abc_317_new_n31_), .B(STATE_REG_1_), .Y(_abc_317_new_n56_));
OR2X2 OR2X2_18 ( .A(_abc_317_new_n61_), .B(_abc_317_new_n21_), .Y(_abc_317_new_n62_));
OR2X2 OR2X2_19 ( .A(_abc_317_new_n62_), .B(_abc_317_new_n38_), .Y(n49));
OR2X2 OR2X2_2 ( .A(_abc_317_new_n16_), .B(STATE_REG_1_), .Y(_abc_317_new_n17_));
OR2X2 OR2X2_20 ( .A(_abc_317_new_n64_), .B(_abc_317_new_n66_), .Y(n53));
OR2X2 OR2X2_21 ( .A(_abc_317_new_n42_), .B(EQL), .Y(_abc_317_new_n68_));
OR2X2 OR2X2_22 ( .A(_abc_317_new_n69_), .B(CONT_EQL), .Y(_abc_317_new_n70_));
OR2X2 OR2X2_3 ( .A(_abc_317_new_n18_), .B(STATE_REG_0_), .Y(_abc_317_new_n19_));
OR2X2 OR2X2_4 ( .A(STATE_REG_1_), .B(STATE_REG_0_), .Y(_abc_317_new_n23_));
OR2X2 OR2X2_5 ( .A(_abc_317_new_n24_), .B(_abc_317_new_n21_), .Y(_abc_317_new_n25_));
OR2X2 OR2X2_6 ( .A(_abc_317_new_n20_), .B(_abc_317_new_n25_), .Y(n36));
OR2X2 OR2X2_7 ( .A(_abc_317_new_n24_), .B(_abc_317_new_n27_), .Y(_abc_317_new_n28_));
OR2X2 OR2X2_8 ( .A(_abc_317_new_n30_), .B(STATE_REG_2_), .Y(_abc_317_new_n31_));
OR2X2 OR2X2_9 ( .A(_abc_317_new_n34_), .B(_abc_317_new_n21_), .Y(_abc_317_new_n35_));


endmodule