`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    12:02:03 10/22/2006 
// Design Name: 
// Module Name:    vgachr 
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
// Simulation plugs exist in this code. Look for "????? SIMULATION PLUG"
//
// Debug plugs exist in this code. Look for "????? DEBUG PLUG"
//
//////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// TERMINAL EMULATOR
//
// Emulates an ADM 3A dumb terminal, with a MITS serial I/O board interface.
// Two ports are emulated:
//
// 0: Control
// 1: Data
//
// The MITS serial card has all of its configuration performed by jumpers on the
// card, which means there is no programming configuration required. The data
// to the terminal is sent out of the data port, while the data from the 
// keyboard, which is not yet implemented, is read from the same data port.
// The parity is ignored on output, and set to 0 on input.
//
// The control register ignores all writes, and returns $80 if the terminal is
// busy, otherwise $00. This is the output ready to send bit. The busy bit
// reflects if the state machine is processing an operation. This is a cheat
// that only works with our emulated terminal/serial board pair, because 
// normally there is no way for the local CPU to know that the remote terminal
// is busy. This can lead to problems in the real world, and it's why Unix
// "termcap" terminal descriptions commonly have waiting periods perscribed
// time consuming operations like screen clear. This emulated terminal is
// "ideal" in that it accounts for all of this automatically, but it does not
// hurt the realisim of the emulation. An application that performs delays based
// on real terminal operations won't be incorrect because it performs a delay,
// but a stupid application that relies on this "smart" implementation might
// fail to run on the real thing.
//
// The ADM 3A terminal emulation is based on "ADM-3A Operators manual" of 1979,
// and "ADM 3A Dumb Terminal Users Reference Manual" of April, 1986, and the
// Unix termcap definition of the terminal.
//
// Several actions of terminals are typically not listed in the documentation,
// and the ADM 3A is no different. That's why termcap exists. The following
// actions were derived from the termcap definition:
//
// 1. The screen clear command also homes the cursor.
// 2. Giving the terminal a right cursor command while at the right side of the
// 3. Screen wraps the cursor around to the right, one line down. If the cursor
//    is at the 80th collumn of the 24th line, it will then scroll.
// 4. Down cursor (line feed) on the 24th line scrolls the screen.
//
// Note that rule (3) causes the terminal to be unable to ever write a character
// to the collumn 80 character of line 24, which is interesting if you are
// writing a full screen editor. The ADM 3A had a switch for this behavior, but
// the user could hardly be expected to open the side panel and flip this switch
// to be able to edit.
//
// The following actions are unknown:
//
// 1. A cursor left command given when the cursor is at the left side of the 
//    screen has unknown effect. According to termcap, it is not to go to the
//    end of the next line up. I have guessed here that it is to simply
//    refuse to move.
// 2. A cursor up command given when the cursor is at the top of the screen
//    has unknown effect. I have guessed here that is is to simply refuse to
//    move.
//
// These are terminal features there are no plans to implement:
//
// 1. ENQ or answerback mode. This appears to require operator setup, the manual
// does not specify any useful default value for the 32 byte answerback, 
// otherwise it might serve as sort of an early plug and play. If there is an
// application out there that uses this, I would put it in.
// 2. Bell (where would it go?).
// 3. Ctrl-N and Ctrl-O keyboard locking and extention port. I'm not sure 
// locking has a good use, nor is all that great an idea.
// 4. Any of the setup modes or features, as operated by the keyboard.
//
// Not implemented in this version:
//
// 1. Reduced intensity mode.
// 2. Graphics mode.
//
// The ADM 3A terminal will pretty much serve as an upward compatible version
// of the ASR-33 teletypes (Western Union Surplus) that were also commonly used
// in the time of the Altair. The ADM 3A was a common replacement for such
// units.
//
// Bugs/problems/issues:
//
// 1. The emulation was occasionally observed to hang. The CPU was still running
// and polling the keyboard, but no reply is received when keys are hit. The rdy
// signal is not coming from the low level keyboard logic.
// 2. Cursor left at left hand side does not refuse to move (see above).
// 3. moving to the 80th collum on line 24 and typing a character does not cause
// the screen to scroll.
// 4. On sign-on, the first two characters are missing the top row. I suspect
// this is due to the start of frame character preloading that is done.
// 5. \ESCGc or set attributes does not work from the keyboard unless you hit
// caps lock or hold the shift key down before and after the escape. It should
// work anytime the upper case G is hit.
//

//
// Terminal height and width
//
`define scnchrs 80 // width
`define scnlins 24 // height

//
// Terminal states
//
`define term_idle    5'h00 // idle
`define term_wrtstd2 5'h01 // write standard character #2
`define term_wrtstd3 5'h02 // write standard character #3
`define term_wrtstd4 5'h03 // write standard character #4
`define term_clear   5'h04 // clear screen and home cursor
`define term_clear2  5'h05 // clear screen and home cursor #2
`define term_clear3  5'h06 // clear screen and home cursor #3
`define term_clear4  5'h07 // clear screen and home cursor #4
`define term_fndstr  5'h08 // find start of current line
`define term_scroll  5'h09 // scroll screen
`define term_scroll1 5'h0a // scroll screen #1
`define term_scroll2 5'h0b // scroll screen #2
`define term_scroll3 5'h0c // scroll screen #3
`define term_scroll4 5'h0d // scroll screen #4
`define term_scroll5 5'h0e // scroll screen #5
`define term_scroll6 5'h0f // scroll screen #6
`define term_esc     5'h10 // escape
`define term_poscur  5'h11 // position cursor
`define term_poscur2 5'h12 // position cursor #2
`define term_attset  5'h13 // set screen attributes

//
// Terminal attribute bits
//
`define attr_blank   5'b00001 // blank
`define attr_blink   5'b00010 // blink
`define attr_reverse 5'b00100 // reverse
`define attr_under   5'b01000 // underline
`define attr_rinten  5'b10000 // reduced intensity

module terminal(addr, data, write, read, select, r, g, b, hsync_n, vsync_n,
                ps2_clk, ps2_data, reset, clock, diag);

   input        addr;     // control reg/data reg address
   inout [7:0]  data;     // CPU data
   input        write;    // CPU write
   input        read;     // CPU read
   input        select;   // controller select
   input        reset;    // CPU reset
   input        clock;    // CPU clock
   output [2:0] r, g, b;  // R,G,B color output buses
   output       hsync_n;  // horizontal sync pulse
   output       vsync_n;  // vertical sync pulse
   input        ps2_clk;  // clock from keyboard
   input        ps2_data; // data from keyboard
   output [7:0] diag;     // diagnostic 8 bit port

   // internal definitions
   reg [4:0]   state;   // terminal state machine
   reg [10:0]  cursor;  // cursor address
   reg [10:0]  tcursor; // cursor temp address
   reg [7:0]   chrdatw; // character write data
   reg         outrdy;  // output ready to send
   reg         wrtchr;  // character ready to write
   reg  [7:0]  datao;   // intermediate for data output
   reg  [7:0]  rowchr;  // row holding character
   reg  [4:0]  curatr;  // current attribute set

   // character map communication bus
   reg  [10:0] cmaddr;  // character map address to read or write
   reg         cmread;  // character map read from address
   reg         cmwrite; // character map write from address
   wire  [7:0] cmdata;  // character map data to be written/read
   wire  [4:0] cmattr;  // character map attributes to be written/read
   reg   [7:0] cmdatai; // character map data to be written
   reg         cmdatae; // character map data enable
   reg   [7:0] cmattri; // character map attribute to be written
   reg         cmattre; // character map attribute enable

   // keyboard communication
   wire   [7:0] scancode;   // key scancode
   wire         parity;     // keyboard parity
   wire         busy;       // busy scanning code
   wire         rdy;        // ready with code
   wire         error;      // scancode error
   reg          scnrdy;     // scancode 
   reg          extkey;     // is extended key
   reg          capslock;   // caps lock toggle
   reg          leftshift;  // state of left shift key
   reg          rightshift; // state of right shift key
   reg          leftctrl;   // state of left control key
   reg          rightctrl;  // state of right control key
   reg          relcod;     // release code active
   reg          extcod;     // extention code active
   wire   [7:0] asciidata;  // output of lookup rom
   wire   [7:0] asciidatau; // output of lookup rom
   reg          clrrdy;     // clear input ready

   // here we put signals out the diagnostic port if required.
   assign diag[0] = busy;
   assign diag[1] = rdy;
   assign diag[2] = scnrdy;
   assign diag[3] = ps2_clk;
   assign diag[4] = ps2_data;
   assign diag[5] = relcod;
   assign diag[6] = leftctrl;
   assign diag[7] = rightctrl;

   // instantiate memory mapped character display
   chrmemmap display(!reset, clock, r, g, b, hsync_n, vsync_n, cmaddr, cmread,
                     cmwrite, cmdata, cmattr, cursor);

   // instantiate ps/2 keyboard. Note that the keyboard decoder only generates
   // codes on release, which has to be changed, since we need both asserts and
   // deasserts.
   ps2_kbd vgai(.clk(clock), .rst(reset), .ps2_clk(ps2_clk), .ps2_data(ps2_data),
                .scancode(scancode), .parity(parity), .busy(busy), .rdy(rdy),
                .error(error));

   // instantiate keyboard scan lookup roms
   scnrom kbdtbl(scancode, asciidata); // lower case
   scnromu kbdtblu(scancode, asciidatau); // lower case

   // process keyboard input state
   always @(posedge clock)
      if (reset) begin // perform reset actions
   
         leftshift <= 0; // clear left shift key state
         rightshift <= 0; // clear right shift key state
         leftctrl <= 0; // clear left control key state
         rightctrl <= 0; // clear right control key state
         capslock <= 0; // clear caps lock state
         relcod <= 0; // clear release status
         scnrdy <= 0; // clear key ready
         extkey <= 0; // is an extended key

   end else begin

      if (rdy) begin

          // if the release code $f0 occurs, set release until the next key occurs
          if (scancode == 8'hf0) relcod <= 1;
          // if the extention code $e0 occurs, set the extention flag
          else if (scancode == 8'he0) extcod <= 1;
          else if (relcod) begin // release

             relcod <= 0; // reset release code
             // reset any extention code
             if (scancode != 8'hf0 && scancode != 8'he0) extcod <= 0;
             // if caps lock is hit, toggle caps lock status
             if (scancode == 8'h58) capslock <= !capslock;
             // process left and right shift key up
             if (scancode == 8'h12) leftshift <= 0; // left up
             if (scancode == 8'h59) rightshift <= 0; // right up
             // process control key up
             if (scancode == 8'h14) begin

                if (extcod) rightctrl <= 0; // right up
                else leftctrl <= 0; // left up

             end

          end else begin // assert

             scnrdy <= 1; // set key is ready
             extkey <= extcod; // set extended status of key
             // reset any extention code
             if (scancode != 8'hf0 && scancode != 8'he0) extcod <= 0;
             // process left and right shift key down
             if (scancode == 8'h12) leftshift <= 1; // left down
             if (scancode == 8'h59) rightshift <= 1; // right down
             // process control key down
             if (scancode == 8'h14) begin

                if (extcod) rightctrl <= 1; // right down
                else leftctrl <= 1; // left down

             end

          end

      end else if (clrrdy) scnrdy <= 0; // clear key ready

   end

   // process terminal emulation
   always @(posedge clock)
      if (reset) begin // reset

         // on reset, we set the state machine to perform a screen clear and
         // home cursor
         // ????? SIMULATION PLUG
         // Don't clear screen, this takes too long in simulation
         state <= `term_clear; // set to clear screen
         // state <= `term_idle; // continue
         cursor <= 0; // set cursor to home
         outrdy <= 0; // set not ready to send
         wrtchr <= 0; // set no character to write
         cmread <= 0; // clear read character
         cmwrite <= 0; // clear write character
         cmdatae <= 0; // no enable character map data
         cmattre <= 0; // no enable character map attributes
         clrrdy <= 0; // set clear ready off
         curatr <= 5'b0; // set no attributes active
                      
   end else begin

      if (write&select) begin // CPU write

         if (addr) begin
      
            chrdatw <= data & 8'h7f; // set character write data without parity
            wrtchr <= 1; // character ready to write
            outrdy <= 0; // remove ready to send
      
         end
      
      end else if (read&select) begin // CPU read
      
         if (addr) begin
      
            // Return decoded keyboard, with shifting. The letter keys, 'a'-'z',
            // can be shifted by the shift keys, but not by caps lock state.
            // The other keys can be shifted by either. This matches PC 
            // behavior, which lets you have cap locks on without shifting the
            // letter keys. The control codes are shifted as a block from `($60)
            // to ~($7e) down to $00. This leaves out \US and \DEL, but these
            // codes get generated by the DEL key.
            if (extkey) begin // its an extended key

               if (scancode == 8'h75) datao <= 8'h0b; // up
               else if (scancode == 8'h6b) datao <= 8'h08; // left
               else if (scancode == 8'h72) datao <= 8'h0a; // down
               else if (scancode == 8'h74) datao <= 8'h0c; // right
               else if (scancode == 8'h6c) datao <= 8'h1e; // home
               else if (scancode == 8'h71) datao <= 8'h1f; // delete (rub out)
    
            end else if (asciidata >= 8'h60 && asciidata <= 8'h7e && 
                         (leftctrl || rightctrl))
               datao <= asciidata & 8'h1f;
            else if (asciidata >= 8'h61 && asciidata <= 8'h7a)
               datao <= leftshift||rightshift||capslock ? asciidatau: asciidata;
            else 
               datao <= leftshift||rightshift ? asciidatau: asciidata;
            clrrdy <= 1; // clear scancode ready f/f
      
         end else 
            datao <= (!outrdy << 7) | (scnrdy << 5); // return ready statuses
      
      end else clrrdy <= 0; // clear keyboard ready signal

      case (state) // run output state
      
         `term_idle: begin // idle waiting for character
      
            // We wait for the cpu cycle to end before running the state machine
            // write procedure. This allows this module to run at full speed, 
            // while the rest of the CPU logic runs slow. The vga logic must run
            // at a fixed speed because it has the display to run, but the rest
            // can be slow to allow debugging.
            if (wrtchr&!(select&read|select&write)) begin 

               // process character after CPU goes away
               if (chrdatw >= 8'h20 && chrdatw != 8'h7f) begin
      
                  // write standard (non-control) character
                  cmaddr <= cursor; // set address at cursor
                  cmdatai <= chrdatw; // place character data to write
                  cmattri <= curatr; // set with current attributes
                  cmdatae <= 1; // enable data to memory
                  cmattre <= 1; // enable attributes to memory
                  state <= `term_wrtstd2; // continue
      
               end else begin // control character
      
                  if (chrdatw == 8'h0a) begin // line down (line feed)
                  
                     if (cursor < 23*80) begin // not at screen end

                        cursor <= cursor+80;
                        wrtchr <= 0; // remove ready to write
                        outrdy <= 1; // set ready to send

                     end else begin

                        tcursor <= 0; // set temp cursor to screen start
                        state <= `term_scroll; // go to scroll screen

                      end
                  
                  end else if (chrdatw == 8'h0b) begin // line up
                  
                     if (cursor >= 80) cursor <= cursor-80;
                     wrtchr <= 0; // remove ready to write
                     outrdy <= 1; // set ready to send
                  
                  end else if (chrdatw == 8'h0d) begin // carriage return
                  
                     tcursor <= 0; // set start of line cursor
                     state <= `term_fndstr; // go to find start of line

                  end else if (chrdatw == 8'h08) begin // character left
                  
                     // if not at home position, back up
                     if (cursor > 0) cursor <= cursor-1;
                     wrtchr <= 0; // remove ready to write
                     outrdy <= 1; // set ready to send

                  end else if (chrdatw == 8'h0c) begin // character right
                  
                     // if not at screen end, go forward
                     if (cursor < 80*24-1) cursor <= cursor+1;
                     wrtchr <= 0; // remove ready to write
                     outrdy <= 1; // set ready to send

                  end else if (chrdatw == 8'h1e) begin // home
                  
                     cursor <= 0; // set to home position
                     wrtchr <= 0; // remove ready to write
                     outrdy <= 1; // set ready to send
                  
                  end else if (chrdatw == 8'h1a) begin // clear screen

                     state <= `term_clear; // go to screen clear
                     wrtchr <= 0; // remove ready to write
                     outrdy <= 1; // set ready to send

                  end else if (chrdatw == 8'h1b) begin // escape

                     wrtchr <= 0; // remove ready to write
                     outrdy <= 1; // set ready to send
                     state <= `term_esc; // go escape handler

                  end else begin // unsupported control character

                     wrtchr <= 0; // remove ready to write
                     outrdy <= 1; // set ready to send

                  end
      
               end
                  
            end
      
         end
      
         `term_wrtstd2: begin // write standard character #2
      
            cmwrite <= 1; // set write to memory
            state <= `term_wrtstd3; // continue
      
         end
      
         `term_wrtstd3: begin // write standard character #3
      
            cmwrite <= 0; // remove write to memory
            state <= `term_wrtstd4; // continue
           
         end

         `term_wrtstd4: begin // write standard character #3
      
            cmdatae <= 0; // release data enable to memory
            cmattre <= 0; // release attribute enable to memory
            outrdy <= 1; // set ready to send
            cursor <= cursor+1; // advance cursor
            wrtchr <= 0; // remove ready to write
            state <= `term_idle; // continue
           
         end
      
         `term_clear: begin // clear screen and home cursor
      
            cmaddr <= 0; // clear buffer address
            cmdatai <= 8'h20; // clear to spaces
            cmattri <= 5'b0; // clear attributes
            cmdatae <= 1; // enable data to memory
            cmattre <= 1; // enable attributes to memory
            state <= `term_clear2; // continue
       
         end
      
         `term_clear2: begin // clear screen and home cursor #2
      
            cmwrite <= 1; // set write to memory
            state <= `term_clear3; // continue
      
         end
      
         `term_clear3: begin // clear screen and home cursor #2
      
            cmwrite <= 0; // reset write to memory
            state <= `term_clear4; // continue
      
         end
      
         `term_clear4: begin // clear screen and home cursor #4
      
            if (cmaddr < `scnchrs*`scnlins) begin
      
               cmaddr <= cmaddr+1; // next character
               // Uncomment the next to put an incrementing pattern instead of
               // spaces.
               //cmdatai <= cmdatai+1;
               state <= `term_clear2; // continue
      
            end else begin // done
      
               outrdy <= 1; // set ready to send
               cursor <= 0; // set cursor to home position
               cmdatae <= 0; // release data enable to memory
               cmattre <= 0; // release attribute enable to memory
               state <= `term_idle; // continue
               
            end
      
         end

         `term_fndstr: begin // find start of current line

            if (tcursor+80 > cursor) begin // found

               cursor <= tcursor; // set cursor to line start
               wrtchr <= 0; // remove ready to write
               outrdy <= 1; // set ready to send
               state <= `term_idle; // continue

            end else tcursor <= tcursor+80; // advance to next line

         end

         `term_scroll: begin // scroll screen up

            // move all data up a line
            if (tcursor < 80*23) begin // scroll up

               cmread <= 1; // set read display
               cmaddr <= tcursor+80; // set address to read
               state <= `term_scroll1; // continue
               
            end else state <= `term_scroll5; // go blank last line

         end

         `term_scroll1: begin // scroll screen up #1

            state <= `term_scroll2; // hold read

         end

         `term_scroll2: begin // scroll screen up #2

            cmdatai <= cmdata; // get data at address
            cmattri <= cmattr; // get attribute at address
            cmread <= 0; // turn off read
            state <= `term_scroll3; // continue

         end

         `term_scroll3: begin // scroll screen up #3

            cmdatae <= 1; // enable data to write
            cmattre <= 1; // enable attribute to write
            cmwrite <= 1; // set to write
            cmaddr <= tcursor;
            state <= `term_scroll4; // continue

         end

         `term_scroll4: begin // scroll screen up #4

            cmwrite <= 0; // turn off write
            cmdatae <= 0; // turn off data enable
            cmattre <= 0; // turn off attribute enable
            tcursor <= tcursor+1; // next address
            state <= `term_scroll; // repeat character move

         end

         `term_scroll5: begin // scroll screen up #5

            // blank out last line
            if (tcursor < 80*24) begin // blank out

               cmdatai <= 8'h20; // set to write spaces
               cmattri <= 0; // set no attribute
               cmdatae <= 1; // enable data to write
               cmattre <= 1; // enable attribute to write
               cmwrite <= 1; // set to write
               cmaddr <= tcursor;
               state <= `term_scroll6; // continue

            end else begin // terminate

               wrtchr <= 0; // remove ready to write
               outrdy <= 1; // set ready to send
               state <= `term_idle; // continue

            end

         end

         `term_scroll6: begin // scroll screen up #6

            cmwrite <= 0; // turn off write
            cmdatae <= 0; // turn off data enable
            cmattre <= 0; // turn off attribute enable
            tcursor <= tcursor+1; // next address
            state <= `term_scroll5; // repeat blank out

         end

         `term_esc: begin // handle escape codes

            // wait for next character
            if (wrtchr&!(select&read|select&write)) begin

               // check its a cursor position, or "\esc="
               if (chrdatw == 8'h3d) begin

                  wrtchr <= 0; // remove ready to write
                  outrdy <= 1; // set ready to send
                  state <= `term_poscur;

               // check its a attribute set, or "\escG"
               end else if (chrdatw == 8'h47) begin

                  wrtchr <= 0; // remove ready to write
                  outrdy <= 1; // set ready to send
                  state <= `term_attset;

               end else begin // invalid sequence, abort

                  wrtchr <= 0; // remove ready to write
                  outrdy <= 1; // set ready to send
                  state <= `term_idle;

               end
    

            end

         end

         `term_poscur: begin // handle cursor direct position

            // wait for next character
            if (wrtchr&!(select&read|select&write)) begin

               rowchr <= chrdatw; // save row character
               wrtchr <= 0; // remove ready to write
               outrdy <= 1; // set ready to send
               state <= `term_poscur2;

            end

         end

         `term_poscur2: begin // handle cursor direct position #2

            // wait for next character
            if (wrtchr&!(select&read|select&write)) begin

               // check row and collumn are valid
               if (rowchr >= 8'h20 && rowchr <= 8'h37 && 
                   chrdatw >= 8'h20 && chrdatw <= 8'h6f)
                  // perform position
                  cursor <= (rowchr-8'h20)*80+(chrdatw-8'h20);
               wrtchr <= 0; // remove ready to write
               outrdy <= 1; // set ready to send
               state <= `term_idle;

            end

         end

         `term_attset: begin // handle attribute set

            // wait for next character
            if (wrtchr&!(select&read|select&write)) begin

               // Process attribute set code. The ADM 3A attributes are arranged
               // so that the bits in characters '0'-'N' correspond to attribute
               // bits. Some of the combinations are "invalid", but these are
               // ones that override each other. For example, blank overrides
               // all others.
               if (chrdatw >= 8'h30 && chrdatw <= 8'h4e) // its '0'-'N'
                  curatr <= chrdatw-8'h30 & 5'b11111;
               wrtchr <= 0; // remove ready to write
               outrdy <= 1; // set ready to send
               state <= `term_idle;

            end

         end
      
         default: state <= 6'bx;
      
      endcase

   end
      
   // Enable drive to character memory
   assign cmdata = cmdatae ? cmdatai: 8'bz;

   // Enable drive to character attributes
   assign cmattr = cmattre ? cmattri: 8'bz;

   // Enable drive for data output
   assign data = read&select ? datao: 8'bz;

endmodule
