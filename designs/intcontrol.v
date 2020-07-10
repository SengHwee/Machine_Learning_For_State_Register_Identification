`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    23:25:07 09/20/2006 
// Design Name: 
// Module Name:    testbench 
// Project Name:                            
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments:                                             
//
//////////////////////////////////////////////////////////////////////////////////

module intcontrol(addr, data, write, read, select, intr, inta, int0, int1, int2,
                  int3, int4, int5, int6, int7, reset, clock);

   input [2:0] addr;   // control register address
   inout [7:0] data;   // CPU data
   input       write;  // CPU write
   input       read;   // CPU read
   input       select; // controller select
   output      intr;   // interrupt request
   input       inta;   // interrupt acknowledge
   input       int0;   // interrupt line 0
   input       int1;   // interrupt line 1
   input       int2;   // interrupt line 2
   input       int3;   // interrupt line 3
   input       int4;   // interrupt line 4
   input       int5;   // interrupt line 5
   input       int6;   // interrupt line 6
   input       int7;   // interrupt line 7
   input       reset;  // CPU reset
   input       clock;  // CPU clock

   reg [7:0] mask;     // interrupt mask register
   reg [7:0] active;   // interrupt active register
   reg [7:0] polarity; // interrupt polarity register
   reg [7:0] edges;    // interrupt edge control
   reg [7:0] vbase;    // vector base
   reg [7:0] intpe;    // positive edge interrupt detection
   reg [7:0] intne;    // negative edge interrupt detection
   reg [7:0] datai; // data from output selector
   reg [3:0] state; // state machine to run vectors
    
   wire [7:0] activep;  // interrupt active pending

   // handle register reads and writes  

   always @(negedge clock)
      if (reset) begin // reset

      mask     <= 8'b0; // clear mask
      active   <= 8'b0; // clear active
      polarity <= 8'b0; // clear polarity
      edges    <= 8'b0; // clear edge
      vbase    <= 8'b0; // clear base
      state    <= 4'b0; // clear state machine

   end else if (write&select) begin // CPU write

      case (addr)

         0: mask     <= data; // set mask register
         2: active   <= data|activep; // set active register
         3: polarity <= data; // set polarity register
         4: edges    <= data; // set edge register
         5: vbase    <= data; // set base register

      endcase

   end else if (read&select) begin // CPU read

      case (addr)

         0: datai <= mask; // get mask register
         // get current line statuses
         1: datai <= { int7, int6, int5, int4, int3, int2, int1, int0 };
         2: datai <= active; // get active register
         3: datai <= polarity; // get polarity register
         4: datai <= edges; // get edge register
         5: datai <= vbase; // get base register

      endcase

   end else if (inta) begin // CPU interrupt acknowledge 

      // run vectoring state machine
      case (state)

         // wait for inta, and assert 1st instruction byte

         0: begin

            datai <= 8'hcd; // place call instruction on datalines
            state <= 1; // advance to low address

         end

         // assert low byte address
         1: begin
         
            // decode priority
            if (active&8'h01)      datai <= 8'h00;
            else if (active&8'h02) datai <= 8'h04;
            else if (active&8'h04) datai <= 8'h08;
            else if (active&8'h08) datai <= 8'h0C;
            else if (active&8'h10) datai <= 8'h10;
            else if (active&8'h20) datai <= 8'h14;
            else if (active&8'h40) datai <= 8'h18;
            else if (active&8'h80) datai <= 8'h1C;
            state <= 2; // advance to high address
         
         end
         
         // assert high address
         2: if (inta) begin
         
            datai <= vbase; // place page to vector
            // reset highest priority interrupt
            if (active&8'h01)      active[0] <= activep[0];
            else if (active&8'h02) active[1] <= activep[1];
            else if (active&8'h04) active[2] <= activep[2];
            else if (active&8'h08) active[3] <= activep[3];
            else if (active&8'h10) active[4] <= activep[4];
            else if (active&8'h20) active[5] <= activep[5];
            else if (active&8'h40) active[6] <= activep[6];
            else if (active&8'h80) active[7] <= activep[7];
            state <= 0; // back to start state
         
         end

      endcase

   end else active <= active|activep; // set active interrupts
      
   // form active interrupt bits
   assign activep = mask & (({ int7, int6, int5, int4, // levels
                               int3, int2, int1, int0 }^polarity & ~edges)|
                           (intpe&polarity&edges)| // positive edges
                           (intne&~polarity&edges)); // negative edges
   
   // form interrupt edges
   always @(posedge int0) intpe[0] <= 1;
   always @(posedge int1) intpe[1] <= 1;
   always @(posedge int2) intpe[2] <= 1;
   always @(posedge int3) intpe[3] <= 1;
   always @(posedge int4) intpe[4] <= 1;
   always @(posedge int5) intpe[5] <= 1;
   always @(posedge int6) intpe[6] <= 1;
   always @(posedge int7) intpe[7] <= 1;
   always @(negedge int0) intne[0] <= 1;
   always @(negedge int1) intne[1] <= 1;
   always @(negedge int2) intne[2] <= 1;
   always @(negedge int3) intne[3] <= 1;
   always @(negedge int4) intne[4] <= 1;
   always @(negedge int5) intne[5] <= 1;
   always @(negedge int6) intne[6] <= 1;
   always @(negedge int7) intne[7] <= 1;

   assign data = read&select|inta ? datai: 8'bz; // enable output data
   assign intr = |active; // request interrupt on any active

endmodule
