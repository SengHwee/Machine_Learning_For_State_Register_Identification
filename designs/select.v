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

module select(addr, data, readio, writeio, select1, select2, select3, select4,
              bootstrap, clock, reset);

   input [15:0] addr;      // CPU address
   inout [7:0]  data;      // CPU data bus
   input        readio;    // I/O read
   input        writeio;   // I/O write
   output       select1;   // select 1
   output       select2;   // select 1
   output       select3;   // select 1
   output       select4;   // select 1
   output       bootstrap; // bootstrap status
   input        clock;     // CPU clock
   input        reset;     // reset
   reg          bootstrap; // bootstrap mode

   reg [7:4]    seladr; // base I/O address of selector
   reg [7:0]    datai; // internal data

   assign selacc = seladr == addr[7:4]; // form selector access
   assign accmain = selacc && (addr[3:1] == 3'b000); // select main
   assign acca = selacc && (addr[3:1] == 3'b001); // select 1
   assign accb = selacc && (addr[3:1] == 3'b010); // select 2
   assign accc = selacc && (addr[3:1] == 3'b011); // select 3
   assign accd = selacc && (addr[3:1] == 3'b100); // select 4

   // Control access to main select unit address. This has to be clocked to
   // activate the address only after the cycle is over.
   always @(posedge clock)
      if (reset) begin

         seladr <= 4'b0; // clear master select
         bootstrap <= 1; // enable bootstrap mode

      end else if (writeio&accmain) begin

         seladr <= data[7:4]; // write master select
         bootstrap <= data[0]; // write bootstrap mode bit

      end else if (readio&accmain) 
         datai <= {seladr, 4'b0}; // read master select
   
   selectone selecta(addr, data, writeio, readio, acca, select1i, reset);
   selectone selectb(addr, data, writeio, readio, accb, select2i, reset);
   selectone selectc(addr, data, writeio, readio, accc, select3, reset);
   selectone selectd(addr, data, writeio, readio, accd, select4, reset);

   assign data = readio&accmain ? datai: 8'bz; // enable output data

   assign select1 = select1i | bootstrap; // enable select 1 via bootstrap
   assign select2 = select2i | bootstrap; // enable select 2 via bootstrap

endmodule
