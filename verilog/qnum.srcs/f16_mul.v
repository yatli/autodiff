//IEEE Floating Point Multiplier (Single Precision)
//Copyright (C) Jonathan P Dawson 2013
//2013-12-12
module f16_mul(
  input     clk,
  input     rst,

  input     [15:0] input_a,
  input     input_a_stb,
  output    input_a_ack,

  input     [15:0] input_b,
  input     input_b_stb,
  output    input_b_ack,

  output    [15:0] output_z,
  output    output_z_stb,
  input     output_z_ack
  );


  reg       s_output_z_stb;
  reg       [15:0] s_output_z;
  reg       s_input_a_ack;
  reg       s_input_b_ack;

  reg       [3:0] state;
  parameter get_a         = 4'd0,
            get_b         = 4'd1,
            unpack        = 4'd2,
            special_cases = 4'd3,
            normalise_a   = 4'd4,
            normalise_b   = 4'd5,
            multiply_0    = 4'd6,
            multiply_1    = 4'd7,
            normalise_1   = 4'd8,
            normalise_2   = 4'd9,
            round         = 4'd10,
            pack          = 4'd11,
            put_z         = 4'd12;

  reg       [15:0] a, b, z;
  reg       [10:0] a_m, b_m, z_m;
  reg       [7:0] a_e, b_e, z_e;
  reg       a_s, b_s, z_s;
  reg       guard, round_bit, sticky;
  reg       [23:0] product;

  wire [23:0] mul_result;
  TopMultiplier multiplier(
    .x_in(a_m),
    .y_in(b_m),
    .result_out(mul_result)
  );

  wire [7:0] adder_ze_1_sum;
  i16_add adder_ze_1(
    .a (z_e[7:0]),
    .b (1),
    .cin (0),
    .sum (adder_ze_1_sum)
  );

  wire [7:0] adder_ae_be_sum;
  i16_add adder_ae_be(
    .a (a_e),
    .b (b_e),
    .cin (0),
    .sum (adder_ae_be_sum)
  );

  wire [26:0] adder_zm_1_sum;
  i32_add adder_zm_1(
    .a (z_m),
    .b (1),
    .cin (0),
    .sum (adder_zm_1_sum)
  );

  wire [7:0] adder_ze_127_sum;
  i16_add adder_ze_127(
    .a (z_e[7:0]),
    .b (127),
    .cin (0),
    .sum (adder_ze_127_sum)
  );



  always @(posedge clk)
  begin

    case(state)

      get_a:
      begin
        s_input_a_ack <= 1;
        if (s_input_a_ack && input_a_stb) begin
          a <= input_a;
          s_input_a_ack <= 0;
          state <= get_b;
        end
      end

      get_b:
      begin
        s_input_b_ack <= 1;
        if (s_input_b_ack && input_b_stb) begin
          b <= input_b;
          s_input_b_ack <= 0;
          state <= unpack;
        end
      end

      unpack:
      begin
        a_m <= a[9 : 0];
        b_m <= b[9 : 0];
        a_e <= a[14 : 10] - 127;
        b_e <= b[14 : 10] - 127;
        a_s <= a[15];
        b_s <= b[15];
        state <= special_cases;
      end

      special_cases:
      begin
        //if a is NaN or b is NaN return NaN 
        if ((a_e == 128 && a_m != 0) || (b_e == 128 && b_m != 0)) begin
          z[15] <= 1;
          z[14:10] <= 255;
          z[9] <= 1;
          z[8:0] <= 0;
          state <= put_z;
        //if a is inf return inf
        end else if (a_e == 128) begin
          z[15] <= a_s ^ b_s;
          z[14:10] <= 255;
          z[9:0] <= 0;
          //if b is zero return NaN
          if (($signed(b_e) == -127) && (b_m == 0)) begin
            z[15] <= 1;
            z[14:10] <= 255;
            z[9] <= 1;
            z[8:0] <= 0;
          end
          state <= put_z;
        //if b is inf return inf
        end else if (b_e == 128) begin
          z[15] <= a_s ^ b_s;
          z[14:10] <= 255;
          z[9:0] <= 0;
          //if a is zero return NaN
          if (($signed(a_e) == -127) && (a_m == 0)) begin
            z[15] <= 1;
            z[14:10] <= 255;
            z[9] <= 1;
            z[8:0] <= 0;
          end
          state <= put_z;
        //if a is zero return zero
        end else if (($signed(a_e) == -127) && (a_m == 0)) begin
          z[15] <= a_s ^ b_s;
          z[14:10] <= 0;
          z[9:0] <= 0;
          state <= put_z;
        //if b is zero return zero
        end else if (($signed(b_e) == -127) && (b_m == 0)) begin
          z[15] <= a_s ^ b_s;
          z[14:10] <= 0;
          z[9:0] <= 0;
          state <= put_z;
        end else begin
          //Denormalised Number
          if ($signed(a_e) == -127) begin
            a_e <= -126;
          end else begin
            a_m[10] <= 1;
          end
          //Denormalised Number
          if ($signed(b_e) == -127) begin
            b_e <= -126;
          end else begin
            b_m[10] <= 1;
          end
          state <= normalise_a;
        end
      end

      normalise_a:
      begin
        if (a_m[10]) begin
          state <= normalise_b;
        end else begin
          a_m <= a_m << 1;
          a_e <= a_e - 1;
        end
      end

      normalise_b:
      begin
        if (b_m[10]) begin
          state <= multiply_0;
        end else begin
          b_m <= b_m << 1;
          b_e <= b_e - 1;
        end
      end

      multiply_0:
      begin
        z_s <= a_s ^ b_s;
        z_e <= adder_ae_be_sum + 1;
        product <= mul_result << 2; // a_m * b_m * 4; 
        state <= multiply_1;
      end

      multiply_1:
      begin
        z_m <= product[23:13];
        guard <= product[12];
        round_bit <= product[11];
        sticky <= (product[10:0] != 0);
        state <= normalise_1;
      end

      normalise_1:
      begin
        if (z_m[10] == 0) begin
          z_e <= z_e - 1;
          z_m <= z_m << 1;
          z_m[0] <= guard;
          guard <= round_bit;
          round_bit <= 0;
        end else begin
          state <= normalise_2;
        end
      end

      normalise_2:
      begin
        if ($signed(z_e) < -126) begin
          z_e <= adder_ze_1_sum;
          z_m <= z_m >> 1;
          guard <= z_m[0];
          round_bit <= guard;
          sticky <= sticky | round_bit;
        end else begin
          state <= round;
        end
      end

      round:
      begin
        if (guard && (round_bit | sticky | z_m[0])) begin
          z_m <= adder_zm_1_sum;
          if (z_m == 11'h7ff) begin
            z_e <=adder_ze_1_sum;
          end
        end
        state <= pack;
      end

      pack:
      begin
        z[9:0] <= z_m[9:0];
        z[14:10] <= adder_ze_127_sum; // z_e[7:0] + 127;
        z[15] <= z_s;
        if ($signed(z_e) == -126 && z_m[10] == 0) begin
          z[14:10] <= 0;
        end
        //if overflow occurs, return inf
        if ($signed(z_e) > 127) begin
          z[9:0] <= 0;
          z[14:10] <= 255;
          z[15] <= z_s;
        end
        state <= put_z;
      end

      put_z:
      begin
        s_output_z_stb <= 1;
        s_output_z <= z;
        if (s_output_z_stb && output_z_ack) begin
          s_output_z_stb <= 0;
          state <= get_a;
        end
      end

    endcase

    if (rst == 1) begin
      state <= get_a;
      s_input_a_ack <= 0;
      s_input_b_ack <= 0;
      s_output_z_stb <= 0;
    end

  end
  assign input_a_ack = s_input_a_ack;
  assign input_b_ack = s_input_b_ack;
  assign output_z_stb = s_output_z_stb;
  assign output_z = s_output_z;

endmodule

