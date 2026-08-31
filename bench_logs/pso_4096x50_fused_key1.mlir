module {
  func.func @main(%arg0: tensor<4096x50xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<4096x50xf32>, %arg3: tensor<4096x50xf32>, %arg4: tensor<4096xf32>, %arg5: tensor<50xf32>, %arg6: tensor<1xf32>, %arg7: tensor<1xi64>) -> (tensor<4096x50xf32>, tensor<4096xf32>, tensor<4096x50xf32>, tensor<4096x50xf32>, tensor<4096xf32>, tensor<50xf32>, tensor<f32>, tensor<4096x50xf32>, tensor<4096xf32>, tensor<4096xf32>) {
    %arg6_r = stablehlo.reshape %arg6 : (tensor<1xf32>) -> tensor<f32>
    %arg7_r = stablehlo.reshape %arg7 : (tensor<1xi64>) -> tensor<i64>
    %0 = stablehlo.constant dense<[[-5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082, -5.119999885559082]]> : tensor<1x50xf32>
    %1 = stablehlo.constant dense<[[5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082, 5.119999885559082]]> : tensor<1x50xf32>
    %2 = "stablehlo.compare"(%arg4, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xi1>
    %3 = stablehlo.reshape %2 : (tensor<4096xi1>) -> tensor<4096x1xi1>
    %5 = "stablehlo.broadcast_in_dim"(%3) {broadcast_dimensions = array<i64: 0, 1>} : (tensor<4096x1xi1>) -> tensor<4096x50xi1>
    %4 = stablehlo.select %5, %arg0, %arg3 : (tensor<4096x50xi1>, tensor<4096x50xf32>, tensor<4096x50xf32>) -> tensor<4096x50xf32>
    %6 = stablehlo.select %2, %arg1, %arg4 : (tensor<4096xi1>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>
    %7 = stablehlo.reshape %arg5 : (tensor<50xf32>) -> tensor<1x50xf32>
    %8 = stablehlo.reshape %arg6_r : (tensor<f32>) -> tensor<1xf32>
    %9 = "stablehlo.concatenate"(%7, %arg0) {dimension = 0 : i64} : (tensor<1x50xf32>, tensor<4096x50xf32>) -> tensor<4097x50xf32>
    %10 = "stablehlo.concatenate"(%8, %arg1) {dimension = 0 : i64} : (tensor<1xf32>, tensor<4096xf32>) -> tensor<4097xf32>
    %11 = stablehlo.reshape %10 : (tensor<4097xf32>) -> tensor<4097xf32>
    %12 = stablehlo.iota dim = 0 : tensor<4097xi64>
    %13 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %14 = stablehlo.constant dense<0> : tensor<i64>
    %26, %27 = "stablehlo.reduce"(%11, %12, %13, %14) ({
      ^bb0(%15: tensor<f32>, %16: tensor<i64>, %17: tensor<f32>, %18: tensor<i64>):
        %19 = "stablehlo.compare"(%15, %17) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %20 = "stablehlo.compare"(%15, %17) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %21 = "stablehlo.compare"(%16, %18) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %22 = stablehlo.and %20, %21 : tensor<i1>
        %23 = stablehlo.or %19, %22 : tensor<i1>
        %24 = stablehlo.select %23, %15, %17 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
        %25 = stablehlo.select %23, %16, %18 : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
        stablehlo.return %24, %25 : tensor<f32>, tensor<i64>
      }) {dimensions = array<i64: 0>} : (tensor<4097xf32>, tensor<4097xi64>, tensor<f32>, tensor<i64>) -> (tensor<f32>, tensor<i64>)
    %28 = stablehlo.reshape %27 : (tensor<i64>) -> tensor<1xi64>
    %29 = stablehlo.constant dense<0> : tensor<i64>
    %30 = "stablehlo.broadcast_in_dim"(%29) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<1xi64>
    %31 = "stablehlo.compare"(%28, %30) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %32 = stablehlo.constant dense<4097> : tensor<i64>
    %33 = "stablehlo.broadcast_in_dim"(%32) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<1xi64>
    %34 = stablehlo.add %28, %33 : tensor<1xi64>
    %35 = stablehlo.select %31, %34, %28 : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %36 = stablehlo.reshape %35 : (tensor<1xi64>) -> tensor<1x1xi64>
    %37 = "stablehlo.gather"(%9, %36) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 50>} : (tensor<4097x50xf32>, tensor<1x1xi64>) -> tensor<1x50xf32>
    %38 = stablehlo.constant dense<0> : tensor<i64>
    %39 = "stablehlo.broadcast_in_dim"(%38) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<1xi64>
    %40 = "stablehlo.compare"(%28, %39) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %41 = stablehlo.constant dense<4097> : tensor<i64>
    %42 = "stablehlo.broadcast_in_dim"(%41) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<1xi64>
    %43 = stablehlo.add %28, %42 : tensor<1xi64>
    %44 = stablehlo.select %40, %43, %28 : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %45 = stablehlo.reshape %44 : (tensor<1xi64>) -> tensor<1x1xi64>
    %46 = "stablehlo.gather"(%10, %45) {dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<4097xf32>, tensor<1x1xi64>) -> tensor<1xf32>
    %47 = stablehlo.reshape %37 : (tensor<1x50xf32>) -> tensor<50xf32>
    %48 = stablehlo.reshape %46 : (tensor<1xf32>) -> tensor<f32>
    %49 = stablehlo.constant dense<0> : tensor<i64>
    %50 = stablehlo.xor %arg7_r, %49 : tensor<i64>
    %51 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %52 = stablehlo.add %50, %51 : tensor<i64>
    %53 = stablehlo.constant dense<30> : tensor<i64>
    %54 = stablehlo.shift_right_logical %52, %53 : tensor<i64>
    %55 = stablehlo.xor %52, %54 : tensor<i64>
    %56 = stablehlo.constant dense<-4658895280553007687> : tensor<i64>
    %57 = stablehlo.multiply %55, %56 : tensor<i64>
    %58 = stablehlo.constant dense<27> : tensor<i64>
    %59 = stablehlo.shift_right_logical %57, %58 : tensor<i64>
    %60 = stablehlo.xor %57, %59 : tensor<i64>
    %61 = stablehlo.constant dense<-7723592293110705685> : tensor<i64>
    %62 = stablehlo.multiply %60, %61 : tensor<i64>
    %63 = stablehlo.constant dense<31> : tensor<i64>
    %64 = stablehlo.shift_right_logical %62, %63 : tensor<i64>
    %65 = stablehlo.xor %62, %64 : tensor<i64>
    %66 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %67 = stablehlo.xor %arg7_r, %66 : tensor<i64>
    %68 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %69 = stablehlo.add %67, %68 : tensor<i64>
    %70 = stablehlo.constant dense<30> : tensor<i64>
    %71 = stablehlo.shift_right_logical %69, %70 : tensor<i64>
    %72 = stablehlo.xor %69, %71 : tensor<i64>
    %73 = stablehlo.constant dense<-4658895280553007687> : tensor<i64>
    %74 = stablehlo.multiply %72, %73 : tensor<i64>
    %75 = stablehlo.constant dense<27> : tensor<i64>
    %76 = stablehlo.shift_right_logical %74, %75 : tensor<i64>
    %77 = stablehlo.xor %74, %76 : tensor<i64>
    %78 = stablehlo.constant dense<-7723592293110705685> : tensor<i64>
    %79 = stablehlo.multiply %77, %78 : tensor<i64>
    %80 = stablehlo.constant dense<31> : tensor<i64>
    %81 = stablehlo.shift_right_logical %79, %80 : tensor<i64>
    %82 = stablehlo.xor %79, %81 : tensor<i64>
    %83 = stablehlo.constant dense<0> : tensor<i64>
    %84 = stablehlo.xor %65, %83 : tensor<i64>
    %85 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %86 = stablehlo.add %84, %85 : tensor<i64>
    %87 = stablehlo.constant dense<30> : tensor<i64>
    %88 = stablehlo.shift_right_logical %86, %87 : tensor<i64>
    %89 = stablehlo.xor %86, %88 : tensor<i64>
    %90 = stablehlo.constant dense<-4658895280553007687> : tensor<i64>
    %91 = stablehlo.multiply %89, %90 : tensor<i64>
    %92 = stablehlo.constant dense<27> : tensor<i64>
    %93 = stablehlo.shift_right_logical %91, %92 : tensor<i64>
    %94 = stablehlo.xor %91, %93 : tensor<i64>
    %95 = stablehlo.constant dense<-7723592293110705685> : tensor<i64>
    %96 = stablehlo.multiply %94, %95 : tensor<i64>
    %97 = stablehlo.constant dense<31> : tensor<i64>
    %98 = stablehlo.shift_right_logical %96, %97 : tensor<i64>
    %99 = stablehlo.xor %96, %98 : tensor<i64>
    %100 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %101 = stablehlo.xor %65, %100 : tensor<i64>
    %102 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %103 = stablehlo.add %101, %102 : tensor<i64>
    %104 = stablehlo.constant dense<30> : tensor<i64>
    %105 = stablehlo.shift_right_logical %103, %104 : tensor<i64>
    %106 = stablehlo.xor %103, %105 : tensor<i64>
    %107 = stablehlo.constant dense<-4658895280553007687> : tensor<i64>
    %108 = stablehlo.multiply %106, %107 : tensor<i64>
    %109 = stablehlo.constant dense<27> : tensor<i64>
    %110 = stablehlo.shift_right_logical %108, %109 : tensor<i64>
    %111 = stablehlo.xor %108, %110 : tensor<i64>
    %112 = stablehlo.constant dense<-7723592293110705685> : tensor<i64>
    %113 = stablehlo.multiply %111, %112 : tensor<i64>
    %114 = stablehlo.constant dense<31> : tensor<i64>
    %115 = stablehlo.shift_right_logical %113, %114 : tensor<i64>
    %116 = stablehlo.xor %113, %115 : tensor<i64>
    %117 = stablehlo.constant dense<0.0> : tensor<f64>
    %118 = stablehlo.constant dense<1.0> : tensor<f64>
    %119 = stablehlo.constant dense<2611923443488327891> : tensor<i64>
    %120 = stablehlo.xor %82, %119 : tensor<i64>
    %121 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %122 = stablehlo.add %120, %121 : tensor<i64>
    %123 = stablehlo.iota dim = 0 : tensor<204800xi64>
    %124 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %125 = "stablehlo.broadcast_in_dim"(%124) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %126 = stablehlo.multiply %123, %125 : tensor<204800xi64>
    %127 = "stablehlo.broadcast_in_dim"(%122) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %128 = stablehlo.add %126, %127 : tensor<204800xi64>
    %129 = stablehlo.constant dense<30> : tensor<i64>
    %130 = "stablehlo.broadcast_in_dim"(%129) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %131 = stablehlo.shift_right_logical %128, %130 : tensor<204800xi64>
    %132 = stablehlo.xor %128, %131 : tensor<204800xi64>
    %133 = stablehlo.constant dense<-4658895280553007687> : tensor<i64>
    %134 = "stablehlo.broadcast_in_dim"(%133) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %135 = stablehlo.multiply %132, %134 : tensor<204800xi64>
    %136 = stablehlo.constant dense<27> : tensor<i64>
    %137 = "stablehlo.broadcast_in_dim"(%136) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %138 = stablehlo.shift_right_logical %135, %137 : tensor<204800xi64>
    %139 = stablehlo.xor %135, %138 : tensor<204800xi64>
    %140 = stablehlo.constant dense<-7723592293110705685> : tensor<i64>
    %141 = "stablehlo.broadcast_in_dim"(%140) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %142 = stablehlo.multiply %139, %141 : tensor<204800xi64>
    %143 = stablehlo.constant dense<31> : tensor<i64>
    %144 = "stablehlo.broadcast_in_dim"(%143) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %145 = stablehlo.shift_right_logical %142, %144 : tensor<204800xi64>
    %146 = stablehlo.xor %142, %145 : tensor<204800xi64>
    %147 = stablehlo.constant dense<11> : tensor<i64>
    %148 = "stablehlo.broadcast_in_dim"(%147) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %149 = stablehlo.shift_right_logical %146, %148 : tensor<204800xi64>
    %150 = stablehlo.convert %149 : (tensor<204800xi64>) -> tensor<204800xf64>
    %151 = stablehlo.constant dense<1.1102230246251565E-16> : tensor<f64>
    %152 = "stablehlo.broadcast_in_dim"(%151) {broadcast_dimensions = array<i64>} : (tensor<f64>) -> tensor<204800xf64>
    %153 = stablehlo.multiply %150, %152 : tensor<204800xf64>
    %154 = stablehlo.reshape %153 : (tensor<204800xf64>) -> tensor<4096x50xf64>
    %155 = "stablehlo.broadcast_in_dim"(%117) {broadcast_dimensions = array<i64>} : (tensor<f64>) -> tensor<4096x50xf64>
    %156 = "stablehlo.broadcast_in_dim"(%118) {broadcast_dimensions = array<i64>} : (tensor<f64>) -> tensor<4096x50xf64>
    %157 = stablehlo.subtract %156, %155 : tensor<4096x50xf64>
    %158 = stablehlo.multiply %154, %157 : tensor<4096x50xf64>
    %159 = stablehlo.add %155, %158 : tensor<4096x50xf64>
    %160 = stablehlo.convert %159 : (tensor<4096x50xf64>) -> tensor<4096x50xf32>
    %161 = stablehlo.constant dense<0.0> : tensor<f64>
    %162 = stablehlo.constant dense<1.0> : tensor<f64>
    %163 = stablehlo.constant dense<2611923443488327891> : tensor<i64>
    %164 = stablehlo.xor %116, %163 : tensor<i64>
    %165 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %166 = stablehlo.add %164, %165 : tensor<i64>
    %167 = stablehlo.iota dim = 0 : tensor<204800xi64>
    %168 = stablehlo.constant dense<-7046029254386353131> : tensor<i64>
    %169 = "stablehlo.broadcast_in_dim"(%168) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %170 = stablehlo.multiply %167, %169 : tensor<204800xi64>
    %171 = "stablehlo.broadcast_in_dim"(%166) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %172 = stablehlo.add %170, %171 : tensor<204800xi64>
    %173 = stablehlo.constant dense<30> : tensor<i64>
    %174 = "stablehlo.broadcast_in_dim"(%173) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %175 = stablehlo.shift_right_logical %172, %174 : tensor<204800xi64>
    %176 = stablehlo.xor %172, %175 : tensor<204800xi64>
    %177 = stablehlo.constant dense<-4658895280553007687> : tensor<i64>
    %178 = "stablehlo.broadcast_in_dim"(%177) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %179 = stablehlo.multiply %176, %178 : tensor<204800xi64>
    %180 = stablehlo.constant dense<27> : tensor<i64>
    %181 = "stablehlo.broadcast_in_dim"(%180) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %182 = stablehlo.shift_right_logical %179, %181 : tensor<204800xi64>
    %183 = stablehlo.xor %179, %182 : tensor<204800xi64>
    %184 = stablehlo.constant dense<-7723592293110705685> : tensor<i64>
    %185 = "stablehlo.broadcast_in_dim"(%184) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %186 = stablehlo.multiply %183, %185 : tensor<204800xi64>
    %187 = stablehlo.constant dense<31> : tensor<i64>
    %188 = "stablehlo.broadcast_in_dim"(%187) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %189 = stablehlo.shift_right_logical %186, %188 : tensor<204800xi64>
    %190 = stablehlo.xor %186, %189 : tensor<204800xi64>
    %191 = stablehlo.constant dense<11> : tensor<i64>
    %192 = "stablehlo.broadcast_in_dim"(%191) {broadcast_dimensions = array<i64>} : (tensor<i64>) -> tensor<204800xi64>
    %193 = stablehlo.shift_right_logical %190, %192 : tensor<204800xi64>
    %194 = stablehlo.convert %193 : (tensor<204800xi64>) -> tensor<204800xf64>
    %195 = stablehlo.constant dense<1.1102230246251565E-16> : tensor<f64>
    %196 = "stablehlo.broadcast_in_dim"(%195) {broadcast_dimensions = array<i64>} : (tensor<f64>) -> tensor<204800xf64>
    %197 = stablehlo.multiply %194, %196 : tensor<204800xf64>
    %198 = stablehlo.reshape %197 : (tensor<204800xf64>) -> tensor<4096x50xf64>
    %199 = "stablehlo.broadcast_in_dim"(%161) {broadcast_dimensions = array<i64>} : (tensor<f64>) -> tensor<4096x50xf64>
    %200 = "stablehlo.broadcast_in_dim"(%162) {broadcast_dimensions = array<i64>} : (tensor<f64>) -> tensor<4096x50xf64>
    %201 = stablehlo.subtract %200, %199 : tensor<4096x50xf64>
    %202 = stablehlo.multiply %198, %201 : tensor<4096x50xf64>
    %203 = stablehlo.add %199, %202 : tensor<4096x50xf64>
    %204 = stablehlo.convert %203 : (tensor<4096x50xf64>) -> tensor<4096x50xf32>
    %205 = stablehlo.constant dense<0.6000000238418579> : tensor<f32>
    %207 = "stablehlo.broadcast_in_dim"(%205) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<4096x50xf32>
    %206 = stablehlo.multiply %207, %arg2 : tensor<4096x50xf32>
    %208 = stablehlo.constant dense<2.5> : tensor<f32>
    %210 = "stablehlo.broadcast_in_dim"(%208) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<4096x50xf32>
    %209 = stablehlo.multiply %210, %204 : tensor<4096x50xf32>
    %211 = stablehlo.subtract %4, %arg0 : tensor<4096x50xf32>
    %212 = stablehlo.multiply %209, %211 : tensor<4096x50xf32>
    %213 = stablehlo.add %206, %212 : tensor<4096x50xf32>
    %214 = stablehlo.constant dense<0.800000011920929> : tensor<f32>
    %216 = "stablehlo.broadcast_in_dim"(%214) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<4096x50xf32>
    %215 = stablehlo.multiply %216, %160 : tensor<4096x50xf32>
    %218 = "stablehlo.broadcast_in_dim"(%47) {broadcast_dimensions = array<i64: 1>} : (tensor<50xf32>) -> tensor<4096x50xf32>
    %217 = stablehlo.subtract %218, %arg0 : tensor<4096x50xf32>
    %219 = stablehlo.multiply %215, %217 : tensor<4096x50xf32>
    %220 = stablehlo.add %213, %219 : tensor<4096x50xf32>
    %221 = stablehlo.add %arg0, %220 : tensor<4096x50xf32>
    %223 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = array<i64: 0, 1>} : (tensor<1x50xf32>) -> tensor<4096x50xf32>
    %222 = stablehlo.maximum %221, %223 : tensor<4096x50xf32>
    %225 = "stablehlo.broadcast_in_dim"(%1) {broadcast_dimensions = array<i64: 0, 1>} : (tensor<1x50xf32>) -> tensor<4096x50xf32>
    %224 = stablehlo.minimum %222, %225 : tensor<4096x50xf32>
    %227 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = array<i64: 0, 1>} : (tensor<1x50xf32>) -> tensor<4096x50xf32>
    %226 = stablehlo.maximum %220, %227 : tensor<4096x50xf32>
    %229 = "stablehlo.broadcast_in_dim"(%1) {broadcast_dimensions = array<i64: 0, 1>} : (tensor<1x50xf32>) -> tensor<4096x50xf32>
    %228 = stablehlo.minimum %226, %229 : tensor<4096x50xf32>
    %230 = stablehlo.constant dense<2.0> : tensor<f32>
    %232 = "stablehlo.broadcast_in_dim"(%230) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<4096x50xf32>
    %231 = stablehlo.power %224, %232 : tensor<4096x50xf32>
    %233 = stablehlo.constant dense<0.0> : tensor<f32>
    %237 = "stablehlo.reduce"(%231, %233) ({
      ^bb0(%235: tensor<f32>, %236: tensor<f32>):
        %234 = stablehlo.add %235, %236 : tensor<f32>
        stablehlo.return %234 : tensor<f32>
      }) {dimensions = array<i64: 1>} : (tensor<4096x50xf32>, tensor<f32>) -> tensor<4096xf32>
    func.return %224, %237, %228, %4, %6, %47, %48, %224, %237, %237 : tensor<4096x50xf32>, tensor<4096xf32>, tensor<4096x50xf32>, tensor<4096x50xf32>, tensor<4096xf32>, tensor<50xf32>, tensor<f32>, tensor<4096x50xf32>, tensor<4096xf32>, tensor<4096xf32>
  }
}
