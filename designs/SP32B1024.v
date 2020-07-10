`timescale 1ns / 1ps

//  Xilinx Single Port Write First RAM
//  This code implements a parameterizable single-port write-first memory where when data
//  is written to the memory, the output reflects the same data being written to the memory.
//  If the output data is not needed during writes or the last read value is desired to be
//  it is suggested to use a No Change as it is more power efficient.
//  If a reset or enable is not necessary, it may be tied off or removed from the code.
//  Modify the parameters for the desired RAM characteristics.

module SP32B1024 (
   output [31:0] Q,
   input CLK,
   input CEN,
   input WEN,
   input [9:0] A,
   input [31:0] D
);


  xilinx_single_port_ram_write_first #(
    .RAM_WIDTH(32),                       // Specify RAM data width
    .RAM_DEPTH(1024),                     // Specify RAM depth (number of entries)
    .INIT_FILE("")                        // Specify name/location of RAM initialization file if using one (leave blank if not)
  ) your_instance_name (
    .addra(A),     // Address bus, width determined from RAM_DEPTH
    .dina(D),       // RAM input data, width determined from RAM_WIDTH
    .clka(CLK),       // Clock
    .wea(WEN),         // Write enable
    .ena(CEN),         // RAM Enable, for additional power savings, disable port when not in use
     .douta(Q)      // RAM output data, width determined from RAM_WIDTH
  );

endmodule
