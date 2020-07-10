////////////////////////////////////////////////////////////////////////////////
// Company:                                                                   //  
// Engineer:       Scott Moore                                                //
//                                                                            //
// Create Date:    11:45:32 09/04/2006                                        // 
// Design Name:                                                               // 
// Module Name:    cpu8080                                                    //
// Project Name:   cpu8080                                                    //
// Target Devices: xc3c200, xc3s1000                                          //
// Tool versions:                                                             //
// Description:                                                               //
//                                                                            //
//     Executes the 8080 instruction set. It is designed to be an internal    //
//     cell. Each of the I/Os are positive logic, and all signals are         //
//     constant with the exception of the data bus. The control signals are   //
//     fully decoded (unlike the orignal 8080), and features read and write   //
//     signals for both memory and I/O space. The I/O space is an 8 bit       //
//     address as in the original 8080. It does NOT echo the lower 8 bits to  //
//     the higher 8 bits, as was the practice in some systems.                //
//                                                                            //
//     Like the original 8080, the interrupt vectoring is fully external. The //
//     the external controller forces a full instruction onto the data bus.   //
//     The sequence begins with the assertion of interrupt request. The CPU   //
//     will then assert interrupt acknowledge, then it will run a special     //
//     read cycle with inta asserted for each cycle of a possibly             //
//     multibyte instruction. This matches the original 8080, which typically //
//     used single byte restart instructions to form a simple interrupt       //
//     controller, but was capable of full vectoring via insertion of a jump, //
//     call or similar instruction.                                           //
//                                                                            //
//     Note that the interrupt vector instruction should branch. This is      //
//     because the PC gets changed by the vector instruction, so if it does   //
//     not branch, it will have skipped a number of bytes after the interrupt //
//     equivalent to the vector instruction. The only instructions that       //
//     should really be used to vector are jmp, rst and call instructions.    //
//     Specifically, rst and call instruction compensate for the pc movement  //
//     by putting the pc unmodified on the stack.                             //
//                                                                            //
//     The memory, I/O and interrupt fetches all obey a simple clocking       //
//     sequence as follows. The CPU uses the positive clock edge to assert    //
//     and sample signals and data. The external logic theoretically uses the //
//     negative edge to check signal assertions and sample data, but it can   //
//     either use the negative edge, or actually be asynronous logic.         //
//                                                                            //
//     A standard read sequence is as follows:                                //
//                                                                            //
//     1. At the positive clock edge, readmem, readio or readint is asserted. //
//     2. At the negative clock edge (or immediately), the external memory    //
//        places data onto the data bus.                                      //
//     3. We hold automatically for one cycle.                                //
//     4. At the next positive clock edge, the data is sampled, and the read  //
//        Signal is deasserted.                                               //
//                                                                            //
//     A standard write sequence is as follows:                               //
//                                                                            //
//     1. At the positive edge, data is asserted on the data bus.             //
//     2. At the next postive clock edge, writemem or writeio is asserted.    //
//     3. At the next positive clock edge, writemem or writeio is deasserted. //
//     4. At the next positive edge, the data is deasserted.                  //
//                                                                            //
// Dependencies:                                                              //
//                                                                            //
// Revision:                                                                  //
// Revision 0.01 - File Created                                               //
// Additional Comments:                                                       //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

//
// CPU states
//

`define cpus_idle     6'h00 // Idle
`define cpus_fetchi   6'h01 // Instruction fetch
`define cpus_fetchi2  6'h02 // Instruction fetch 2
`define cpus_fetchi3  6'h03 // Instruction fetch 3
`define cpus_fetchi4  6'h04 // Instruction fetch 4
`define cpus_halt     6'h05 // Halt (wait for interrupt)
`define cpus_alucb    6'h06 // alu cycleback
`define cpus_indcb    6'h07 // inr/dcr cycleback
`define cpus_movmtbc  6'h08 // Move memory to bc
`define cpus_movmtde  6'h09 // Move memory to de
`define cpus_movmthl  6'h0a // Move memory to hl
`define cpus_movmtsp  6'h0b // Move memory to sp
`define cpus_lhld     6'h0c // LHLD
`define cpus_jmp      6'h0d // JMP
`define cpus_write    6'h0e // write byte
`define cpus_write2   6'h0f // write byte #2
`define cpus_write3   6'h10 // write byte #3
`define cpus_write4   6'h11 // write byte #4
`define cpus_read     6'h12 // read byte
`define cpus_read2    6'h13 // read byte #2
`define cpus_read3    6'h14 // read byte #3
`define cpus_pop      6'h15 // POP completion
`define cpus_in       6'h16 // IN
`define cpus_in2      6'h17 // IN #2
`define cpus_in3      6'h18 // IN #3
`define cpus_out      6'h19 // OUT
`define cpus_out2     6'h1a // OUT #2
`define cpus_out3     6'h1b // OUT #3
`define cpus_out4     6'h1c // OUT #4
`define cpus_movtr    6'h1d // move to register
`define cpus_movrtw   6'h1e // move read to write
`define cpus_movrtwa  6'h1f // move read to write address
`define cpus_movrtra  6'h20 // move read to read address
`define cpus_accimm   6'h21 // accumulator immediate operations
`define cpus_daa      6'h22 // DAA completion

//
// Register numbers
//

`define reg_b 3'b000 // B
`define reg_c 3'b001 // C
`define reg_d 3'b010 // D
`define reg_e 3'b011 // E
`define reg_h 3'b100 // H
`define reg_l 3'b101 // L
`define reg_m 3'b110 // M
`define reg_a 3'b111 // A

//
// ALU operations
//

`define aluop_add 3'b000 // add
`define aluop_adc 3'b001 // add with carry in
`define aluop_sub 3'b010 // subtract
`define aluop_sbb 3'b011 // subtract with borrow in
`define aluop_and 3'b100 // and
`define aluop_xor 3'b101 // xor
`define aluop_or  3'b110 // or
`define aluop_cmp 3'b111 // compare

//
// State macros
//
`define mac_writebyte  1  // write a byte
`define mac_readbtoreg 2  // read a byte, place in register
`define mac_readdtobc  4  // read double byte to BC
`define mac_readdtode  6  // read double byte to DE
`define mac_readdtohl  8 // read double byte to HL
`define mac_readdtosp  10 // read double byte to SP
`define mac_readbmtw   12 // read byte and move to write
`define mac_readbmtr   14 // read byte and move to register
`define mac_sta        16 // STA
`define mac_lda        20 // LDA
`define mac_shld       25 // SHLD
`define mac_lhld       30 // LHLD
`define mac_writedbyte 36 // write double byte
`define mac_pop        38 // POP
`define mac_xthl       40 // XTHL
`define mac_accimm     44 // accumulator immediate
`define mac_jmp        45 // JMP
`define mac_call       47 // CALL
`define mac_in         51 // IN
`define mac_out        52 // OUT
`define mac_rst        53 // RST

module alu(res, opra, oprb, cin, cout, zout, sout, parity, auxcar, sel);

   input  [7:0] opra;   // Input A
   input  [7:0] oprb;   // Input B
   input        cin;    // Carry in
   output       cout;   // Carry out
   output       zout;   // Zero out
   output       sout;   // Sign out
   output       parity; // parity
   output       auxcar; // auxiliary carry
   input  [2:0] sel;    // Operation select
   output [7:0] res;    // Result of alu operation
   
   reg       cout;   // Carry out
   reg       zout;   // Zero out
   reg       sout;   // sign out
   reg       parity; // parity
   reg       auxcar; // auxiliary carry
   reg [7:0] resi;   // Result of alu operation intermediate
   reg [7:0] res;    // Result of alu operation

   always @(opra, oprb, cin, sel, res, resi) begin

      case (sel)
      
         `aluop_add: begin // add

            { cout, resi } = opra+oprb; // find result and carry
//            auxcar = ((opra[3:0]+oprb[3:0]) >> 4) & 1'b1; // find auxiliary carry
//			if ((opra[3:0]+oprb[3:0])>>4) auxcar=1'b1; else auxcar=1'b0;
            auxcar = (((opra[3:0]+oprb[3:0]) >> 4) & 1'b1) ? 1'b1 : 1'b0 ; // find auxiliary carry


         end
         `aluop_adc: begin // adc

            { cout, resi } = opra+oprb+cin; // find result and carry
            auxcar = (((opra[3:0]+oprb[3:0]+cin) >> 4) & 1'b1) ? 1'b1 : 1'b0; // find auxiliary carry

         end
         `aluop_sub, `aluop_cmp: begin // sub/cmp

            { cout, resi } = opra-oprb; // find result and carry
            auxcar = (((opra[3:0]-oprb[3:0]) >> 4) & 1'b1) ? 1'b1 : 1'b0; // find auxiliary borrow

         end
         `aluop_sbb: begin // sbb

            { cout, resi } = opra-oprb-cin; // find result and carry
            auxcar = (((opra[3:0]-oprb[3:0]-cin >> 4)) & 1'b1) ? 1'b1 : 1'b0; // find auxiliary borrow 

         end
         `aluop_and: begin // ana

            { cout, resi } = {1'b0, opra&oprb}; // find result and carry
            auxcar = 1'b0; // clear auxillary carry

          end
         `aluop_xor: begin // xra

            { cout, resi } = {1'b0, opra^oprb}; // find result and carry
            auxcar = 1'b0; // clear auxillary carry

         end
         `aluop_or:  begin // ora

            { cout, resi } = {1'b0, opra|oprb}; // find result and carry
            auxcar = 1'b0; // clear auxillary carry

         end
                     
      endcase

      if (sel != `aluop_cmp) res = resi; else res = opra;
      zout <= ~|resi; // set zero flag from result
      sout <= resi[7]; // set sign flag from result
      parity <= ~^resi; // set parity flag from result

   end

endmodule