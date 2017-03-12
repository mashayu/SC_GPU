#include "decoder.h"

// #define LOG
// #define LOG_P
//#define CL_TEST
//#define CPU
#define OPENCL
#define MAX_SOURCE_SIZE (0x100000)
Decoder::Decoder(int n, int k, set<int> _frozen, double sd) :
m(log2(n)), N(n), K(k), dispersion(sd)
{
	frozen = _frozen;
	// std::cout << "m = " << m << std::endl;
#ifdef CPU
	P = new double**[m + 1];
	C = new bool*[m + 1];
		S = new double*[m + 1];
	
#endif

#ifdef OPENCL

		buffer_S.reserve(m + 1);
		buffer_C0.reserve(m + 1);
		buffer_C1.reserve(m + 1);
	//get all platforms (drivers)
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0){
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	cl::Platform default_platform = all_platforms[1];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	//get default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0){
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	//creating context
	cl::Context context({ default_device});

	
#endif

	for (int i = 0; i <= m; ++i) {
#ifdef OPENCL
		buffer_S.push_back(*new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * (1 << (m - i))));
		buffer_C0.push_back(*new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(bool) * (1 << (m - i)))); 
		buffer_C1.push_back(*new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(bool) * (1 << (m - i)))); 
#endif
#ifdef CPU
		P[i] = new double*[(int)pow(2, m - i)];
		S[i] = new double[(int)pow(2, m - i)];
		C[i] = new bool[(int)pow(2, m - i)];
		for (int j = 0; j < (int)pow(2, m - i); ++j) {
			//C[i][j] = new bool[2];
			P[i][j] = new double[2];
			memset(P[i][j], 0, sizeof(double));
			//memset(C[i][j], 0, sizeof(bool)/* * 2*/);
		}
#endif
	}
#ifdef OPENCL

		char* source_str;
		size_t source_size;

		FILE *fp;
		fp = fopen("buxus.cl", "r");
		if (!fp) {
			fprintf(stderr, "Failed to load kernel.\n");
			exit(1);
		}
		source_str = (char*)malloc(MAX_SOURCE_SIZE);
		source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
		sources.push_back({ source_str, source_size });
		fclose(fp);
		//Build program
		program = cl::Program(context, sources);
		if (program.build({ default_device }) != CL_SUCCESS){
			std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
			exit(1);
		}

		queue = cl::CommandQueue(context, default_device);


		y_buf = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * n);


		kernel_S = cl::Kernel(program, "func");
		kernel_softCombine = cl::Kernel(program, "SoftCombine");
		kernel_softxor = cl::Kernel(program, "SoftXOR");
		kernel_xor = cl::Kernel(program, "XOR");
		kernel_wke = cl::Kernel(program, "wierdest_kernel_ever");
		kernel_kew = cl::Kernel(program, "kernel_even_wierder");
		Q_GPU = cl::Kernel(program, "Q");
		P_GPU = cl::Kernel(program, "P_GPU");
		kernel_updC = cl::Kernel(program, "updC");

		test_s = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * n);
#endif


		

	//frozen = new bool[N];
	//memset(frozen, 0, sizeof(bool)*N);
	/*for (int i = N - 1; i >= K; --i) {
		// std::cout << indeces[i] << " " << shuffle[indeces[i]] << " " << indeces[shuffle[i]] << std::endl;
		frozen[indeces[i]] = true;
	}*/
	// std::cout << "frozen positions: ";
	// for (int i = 0; i < N; ++i) {
	//   std::cout << frozen[i] << " ";
	// }
	// std::cout << std::endl;
}

/*Decoder::~Decoder() {
	//delete[] frozen;
	for (int i = 0; i <= m; ++i) {
		delete[] P[i];
		for (int j = 0; j < N; ++j) {

			//delete[] C[i][j];
		}
		delete[] P[i];
		delete[] C[i];
	}
	delete[] P;
	delete[] C;
}*/

void Decoder::decode(double *y, bool *c) {

#ifdef OPENCL
	double *test = new double[N];
	double *test2 = new double[N];
	//for (int u = 0; u < N; u++)
		//test2[u] = y[u];
	queue.enqueueWriteBuffer(y_buf, CL_FALSE, 0, sizeof(double) * N, y);
//	queue.enqueueReadBuffer(y_buf, CL_FALSE, 0, sizeof(double) * N, test);
	kernel_S.setArg(0, y_buf);
	kernel_S.setArg(1, buffer_S.at(0));
	kernel_S.setArg(2, dispersion);
	kernel_S.setArg(3, N);
	queue.enqueueNDRangeKernel(kernel_S, cl::NullRange, cl::NDRange(N), cl::NullRange);
#ifdef CL_TEST
	queue.enqueueReadBuffer(buffer_S.at(0), CL_TRUE, 0, sizeof(double) * N, test);
	for (int i = 0; i < N; i++)
		cout << test[i] << " ";
	cout << endl;
#endif
#endif

#ifdef CPU
	for (int beta = 0; beta < N; ++beta) {

		//P[0][beta][0] = exp(-pow((y[beta] - 1), 2) / (2 * dispersion)) / sqrt(2 * M_PI * dispersion);
		//P[0][beta][1] = exp(-pow((y[beta] + 1), 2) / (2 * dispersion)) / sqrt(2 * M_PI * dispersion);

		S[0][beta] = 2 * y[beta] / dispersion;
	}
#endif

	for (int fi = 0; fi < N; ++fi) {
#ifdef OPENCL
		recursivelyCalcS_GPU(m, fi);
		//IterativelyCalcS(m, fi);
#endif
#ifdef CPU
		IterativelyCalcS_CPU(m, fi);
		//recursivelyCalcS(m, fi);
#endif
		//recursivelyCalcP(m, fi);
		int fm2 = fi % 2;

#ifdef LOG_P
		for (int i = 0; i <= m; ++i) {
			std::cout << "P[" << i << "][.][0] : ";
			for (int j = 0; j < (1 << (m - i)); ++j)
				std::cout << P[i][j][0] << " ";
			std::cout << std::endl;
			std::cout << "P[" << i << "][.][1] : ";
			for (int j = 0; j < (1 << (m - i)); ++j)
				std::cout << P[i][j][1] << " ";
			std::cout << std::endl;
		}
#endif
#ifdef LOG
		std::cout << "fi " << fi << std::endl;
		std::cout << "CHOOSING BIT: " << P[m][0][0] << " vs " << P[m][0][1] << std::endl;
		if (frozen[fi]) {
			std::cout << "FROZEN BIT!!!" << std::endl;
			if (P[m][0][0] < P[m][0][1])
				std::cout << "FROZEN FAIL!!!" << std::endl;
			C[m][0][fm2] = false;
		}
		else {
			if (P[m][0][0] > P[m][0][1]) {
				std::cout << "CHOOSE ZERO!!!" << std::endl;
				C[m][0][fm2] = false;
			}
			else {
				std::cout << "CHOOSE ONE!!!" << std::endl;
				C[m][0][fm2] = true;
			}
		}
#else
		if (frozen.find(fi) != frozen.end()) {
#ifdef OPENCL
			bool *testC = new bool[N];
			cl::Buffer bufCW;
			if (fm2 == 0)
				bufCW = buffer_C0.at(m);
			else 
				bufCW = buffer_C1.at(m);
			kernel_wke.setArg(0, bufCW);
			queue.enqueueNDRangeKernel(kernel_wke, cl::NullRange, cl::NDRange(1), cl::NullRange);
#ifdef CL_TEST
			queue.enqueueReadBuffer(bufCW, CL_TRUE, 0, sizeof(bool) * 1, testC);
#endif
#endif

#ifdef CPU
		//	C[m][0][fm2] = false;
			C[m][0] = false;
#endif
		}
		else {
#ifdef OPENCL
			double *testS = new double[1];
			bool *testC = new bool[N];
			cl::Buffer bufCW;
			if (fm2 == 0)
				bufCW = buffer_C0.at(m);
			else
				bufCW = buffer_C1.at(m);
			//cl::Buffer bufCW = buffer_C.at(0);
			cl::Buffer bufS = buffer_S.at(m);

			kernel_kew.setArg(0, bufCW);
			kernel_kew.setArg(1, bufS);
			kernel_kew.setArg(2, fm2);

						
			queue.enqueueNDRangeKernel(kernel_kew, cl::NullRange, cl::NDRange(1), cl::NullRange);
#ifdef CL_TEST
			queue.enqueueReadBuffer(bufS, CL_TRUE, 0, sizeof(double) * 1, testS);
			queue.enqueueReadBuffer(bufCW, CL_TRUE, 0, sizeof(bool) * 1, testC);
#endif

#endif

#ifdef CPU
			if (S[m][0] > 0) {
			//C[m][0][fm2] = false;
			C[m][0] = false;
			}
			else {
				//C[m][0][fm2] = true;
				C[m][0] = true;
			}
#endif
		}
#endif
		if (fm2 == 1){
#ifdef OPENCL
			recursivelyUpdateC_GPU(m, fi);
			//IterativelyUpdateC(m, fi);
#endif
#ifdef CPU
			//recursivelyUpdateC(m, fi);
			IterativelyUpdateC_CPU(m, fi);
#endif
		}
	}
#ifdef OPENCL
	queue.enqueueReadBuffer(buffer_C0.at(0), CL_FALSE, 0, sizeof(bool) * N, c);
#endif


#ifdef CPU
	for (int i = 0; i < N; ++i) {
		//c[i] = C[0][i][0];
		c[i] = C[0][i];
	}

#endif
}

inline double Decoder::W(double y, double x) {
	
	return 1 / (1 + exp((-1)*x * 2 * y / (dispersion)));

}

/*void Decoder::recursivelyCalcP(int lambda, int fi) {
	if (lambda == 0)
		return;
	int fm2 = fi % 2;
	if (fm2 == 0)
		recursivelyCalcP(lambda - 1, fi / 2);
	double maximum = 0;
	for (int beta = 0; beta < (1 << (m - lambda)); ++beta) {
		if (fm2 == 0) {
			P[lambda][beta][0] = P[lambda - 1][2 * beta][0] * P[lambda - 1][2 * beta + 1][0];
			P[lambda][beta][0] += P[lambda - 1][2 * beta][1] * P[lambda - 1][2 * beta + 1][1];
			P[lambda][beta][1] = P[lambda - 1][2 * beta][1] * P[lambda - 1][2 * beta + 1][0];
			P[lambda][beta][1] += P[lambda - 1][2 * beta][0] * P[lambda - 1][2 * beta + 1][1];
			if (maximum < P[lambda][beta][0]) maximum = P[lambda][beta][0];
			if (maximum < P[lambda][beta][1]) maximum = P[lambda][beta][1];
		}
		else {
			bool u = (C[lambda][beta][0]) ? 1 : 0;
			P[lambda][beta][0] = P[lambda - 1][2 * beta][u] * P[lambda - 1][2 * beta + 1][0];
			P[lambda][beta][1] = P[lambda - 1][2 * beta][u ^ 1] * P[lambda - 1][2 * beta + 1][1];
			if (maximum < P[lambda][beta][0]) maximum = P[lambda][beta][0];
			if (maximum < P[lambda][beta][1]) maximum = P[lambda][beta][1];
		}
	}
	if (maximum > 0) {
		for (int beta = 0; beta < (1 << (m - lambda)); ++beta) {
			P[lambda][beta][0] /= maximum;
			P[lambda][beta][1] /= maximum;
		}
	}
}*/

/*void Decoder::recursivelyUpdateC(int lambda, int fi) {
	//assert(fi % 2 == 1);
	int psi = fi / 2;
	int psm2 = psi % 2;
	for (int beta = 0; beta < (1 << (m - lambda)); ++beta) {
		C[lambda - 1][2 * beta][psm2] = C[lambda][beta][0] != C[lambda][beta][1];
		C[lambda - 1][2 * beta + 1][psm2] = C[lambda][beta][1];
	}
	if (psm2 == 1)
		recursivelyUpdateC(lambda - 1, psi);
}*/


void Decoder::recursivelyUpdateC_GPU(int lambda, int fi) {
	//assert(fi % 2 == 1);
	cl::Buffer C1 = buffer_C1.at(lambda);
	cl::Buffer C0 = buffer_C0.at(lambda);
	cl::Buffer Ñ_;
	int psi = fi / 2;
	int psm2 = psi % 2;
	int N1 = (1 << (m - lambda));
	if (psm2 == 0)
		Ñ_ = buffer_C0.at(lambda-1);
	else 
		Ñ_ = buffer_C1.at(lambda-1);
	kernel_updC.setArg(0, Ñ_);
	kernel_updC.setArg(1, C0);
	kernel_updC.setArg(2, C1);
	kernel_updC.setArg(3, N1);
	queue.enqueueNDRangeKernel(kernel_updC, cl::NullRange, cl::NDRange(N1), cl::NullRange);
	if (psm2 == 1)
		recursivelyUpdateC_GPU(lambda - 1, psi);
}


double Q(double a, double b) {
	if ((a < 0 && b < 0) || (a>0 && b>0))
		return (abs(a) < abs(b)) ? abs(a) : abs(b);
	return (abs(a) < abs(b)) ? abs(a)*(-1) : abs(b)*(-1);
}

/*void Decoder::recursivelyCalcS(int lambda, int phi) {
	if (lambda == 0) return;
	if (phi % 2 == 0) recursivelyCalcS(lambda - 1, phi / 2);
	//double* S_lambda = getArray_S(lambda, l);
	//double* S_prelambda = getArray_S(lambda - 1, l); // this is for copying
	//getD(l, lambda) = getD(l, lambda - 1);
	for (int beta = 0; beta < pow(2, m - lambda); beta++) {
		if (phi % 2 == 0) {
			S[lambda][beta] = Q(S[lambda - 1][2 * beta], S[lambda - 1][2 * beta+1]);
		}
		else{
			if (C[lambda][beta][0] == 0) {
				S[lambda][beta] = S[lambda - 1][2 * beta] + S[lambda - 1][2 * beta + 1];
			}
			else {

				S[lambda][beta] = S[lambda - 1][2 * beta + 1] - S[lambda - 1][2 * beta];
			}
		}
	}
}*/


void Decoder::recursivelyCalcS_GPU(int lambda, int phi) {
	
	if (lambda == 0) return;
	if (phi % 2 == 0) recursivelyCalcS_GPU(lambda - 1, phi / 2);
	//double* S_lambda = getArray_S(lambda, l);
	//double* S_prelambda = getArray_S(lambda - 1, l); // this is for copying
	//getD(l, lambda) = getD(l, lambda - 1);
	cl::Buffer S_ = buffer_S.at(lambda);
	cl::Buffer Ñ_ = buffer_C0.at(lambda);
	cl::Buffer S1 = buffer_S.at(lambda - 1);
	unsigned N1 = 1 << (m - lambda);
	double *test = new double[N1];
	if (phi % 2 == 0) {
		Q_GPU.setArg(0, S_);
		Q_GPU.setArg(1, S1);
		Q_GPU.setArg(2, N1);
		queue.enqueueNDRangeKernel(Q_GPU, cl::NullRange, cl::NDRange(N1), cl::NullRange);
#ifdef CL_TEST
		queue.enqueueReadBuffer(S_, CL_TRUE, 0, sizeof(double) * N1, test);
		for (int i = 0; i < 8; i++)
			cout << test[i] << " ";
		cout << endl;
#endif
	}
	
	else {
		P_GPU.setArg(0, Ñ_);
		P_GPU.setArg(1, S_);
		P_GPU.setArg(2, S1);
		P_GPU.setArg(3, N1);
		queue.enqueueNDRangeKernel(P_GPU, cl::NullRange, cl::NDRange(N1), cl::NullRange);
#ifdef CL_TEST
		queue.enqueueReadBuffer(S_, CL_TRUE, 0, sizeof(double) * N1, test);
		for (int i = 0; i < N; i++)
			cout << test[i] << " ";
		cout << endl;
#endif
	}
}


void Decoder::IterativelyCalcS(unsigned lambda, int phi
	)
{
	int Depth = 0;
	while (phi % (int)(pow(2,Depth))) Depth++;
	unsigned lambda0 = lambda - Depth;
	/*if (lambda0 == 0){
		MType* S = new MType[1 << (m_LogLength - lambda0)];

		queue.enqueueReadBuffer(CGetArrayPointer_S(lambda0, pIndexArray), CL_TRUE, 0, sizeof(TV_TYPE_S) * (1 << (m_LogLength - lambda0)), S_pin);
		ptemp = &GetArrayPointer_S(lambda0, pIndexArray);
		for (int i = 0; i < (1 << (m_LogLength - lambda)); i++)
			S[i] = S_pin[i];
		return S; //GetArrayPointer_S(lambda0, pIndexArray);
	}*/
	cl::Buffer S1 = buffer_S.at(lambda0 - 1);
 	unsigned N1 = 1 << (m - lambda0);

#ifdef CL_TEST
	MType *Sq = new MType[1 << (m_LogLength - lambda0 + 1)];
	queue.enqueueReadBuffer(M_l_1, CL_TRUE, 0, sizeof(MType) * (1 << (m_LogLength - lambda0 + 1)), Sq);
#endif

	//unsigned N = (m_BitsPerBlock << (m_LogLength - lambda0));
  	cl::Buffer S_;
	//cl::Buffer *pM_l;
	if ((phi>>Depth)%2==0)
	{
		S_ = buffer_S.at(lambda0);
#ifdef CL_TEST
		MType* Sr = new MType[1 << (m_LogLength - lambda0)];
		queue.enqueueReadBuffer(M_l, CL_TRUE, 0, sizeof(MType) * (1 << (m_LogLength - lambda0)), Sr);
#endif
		cl::Buffer C_l = buffer_C0.at(lambda0);

#ifdef CL_TEST
		tBit* C_ = new tBit[1 << (m_LogLength - lambda0)];
		queue.enqueueReadBuffer(C_l, CL_TRUE, 0, sizeof(tBit) * (1 << (m_LogLength - lambda0)), C_);
#endif

		kernel_softCombine.setArg(0, S_);
		kernel_softCombine.setArg(1, S1);
		kernel_softCombine.setArg(2, C_l);
		kernel_softCombine.setArg(3, N1);
		//SoftCombine(M_l, M_l_1, C_l, N);

		double *test = new double[N1];
			queue.enqueueNDRangeKernel(kernel_softCombine, cl::NullRange, cl::NDRange(N1), cl::NullRange);
#ifdef CL_TEST
			queue.enqueueReadBuffer(S_, CL_TRUE, 0, sizeof(double) * N, test);
#endif
		S1 = buffer_S.at(lambda0);
#ifdef CL_TEST
		MType* St = new MType[1 << (m_LogLength - lambda0)];
		queue.enqueueReadBuffer(M_l_1, CL_TRUE, 0, sizeof(MType) * (1 << (m_LogLength - lambda0)), St);

		MType* Sy = new MType[1 << (m_LogLength - lambda0)];
		queue.enqueueReadBuffer(M_l, CL_TRUE, 0, sizeof(MType) * (1 << (m_LogLength - lambda0)), Sy);
#endif
		lambda0++;
		N1 >>= 1;
	};
	//MType* Sw = new MType[1 << (m_LogLength - lambda0)];
	for (lambda0; lambda0 <= lambda; lambda0++)
	{
		//do the calculation
		S_ = buffer_S.at(lambda0);
#ifdef CL_TEST
		queue.enqueueReadBuffer(M_l, CL_TRUE, 0, sizeof(MType) * (1 << (m_LogLength - lambda0)), Sw);
#endif
		kernel_softxor.setArg(0, S_);
		kernel_softxor.setArg(1, S1);
		kernel_softxor.setArg(2, N1);

		queue.enqueueNDRangeKernel(kernel_softxor, cl::NullRange, cl::NDRange(N), cl::NullRange);
		//if (N == 512)
		//	TEST++;
		//SoftXOR(M_l, M_l_1, M_l_1 + N, N);
#ifdef CL_TEST
		queue.enqueueReadBuffer(M_l, CL_TRUE, 0, sizeof(MType) * (1 << (m_LogLength - lambda0)), Sw);
#endif
		S1 = buffer_S.at(lambda0);
		N1 >>= 1;
	};
/*	MType* S = new MType[1 << (m_LogLength - lambda)];
	ptemp = &M_l;

	queue.enqueueReadBuffer(M_l, CL_TRUE, 0, sizeof(MType) * (1 << (m_LogLength - lambda)), S_pin);
	for (int i = 0; i < (1 << (m_LogLength - lambda)); i++)
		S[i] = S_pin[i];

//	return S;//return M_l;*/
};

void Decoder::IterativelyUpdateC(unsigned lambda, unsigned phi)
{
	int Depth = 0;
	while ((phi+1) % (2 ^ Depth)) Depth++;
	unsigned lambda0 = lambda - Depth;
	//cl::Buffer pC0 = buffer_C.at(lambda);

#ifdef CL_TEST
	tBit* C0 = new tBit[1 << (m_LogLength - lambda)];
	queue.enqueueReadBuffer(CGetArrayPointer_C(lambda, pIndexArray, 0), CL_TRUE, 0, sizeof(tBit) * (1 << (m_LogLength - lambda)), C0);
#endif // CL_TEST

	//write everything to its ultimate destination
	cl::Buffer pC_Out = buffer_C0.at(lambda0);

#ifdef CL_TEST
	tBit* C1 = new tBit[1 << (m_LogLength - lambda0)];
	queue.enqueueReadBuffer(CGetArrayPointer_C(lambda0, pIndexArray, 0), CL_TRUE, 0, sizeof(tBit) * (1 << (m_LogLength - lambda0)), C1);
#endif // CL_TEST


	unsigned N_ = (1 << (m - lambda));




	unsigned FinalLength = N_ << Depth;


	unsigned N1 = FinalLength - N_;
	unsigned N2 = FinalLength - 2 * N_;

	/*kernel_xor.setArg(0, pC_Out);
	kernel_xor.setArg(1, pC0);
	kernel_xor.setArg(2, N1);
	kernel_xor.setArg(3, N2);
	kernel_xor.setArg(4, N);
	//SoftCombine(M_l, M_l_1, C_l, N);


	queue.enqueueNDRangeKernel(kernel_xor, cl::NullRange, cl::NDRange(N), cl::NullRange);
	if (N == 512)
		TEST++;*/

	//queue.enqueueReadBuffer(pC_Out, CL_TRUE, 0, sizeof(tBit) * (1 << (m_LogLength - lambda0)), C1);

	//XOR(pC_Out, pC0, pC1, N);
	//lambda--;

	bool *test = new bool[N];
	for (lambda; lambda > lambda0; lambda--)
	{

		const cl::Buffer pC0 = buffer_C0.at(lambda);
		//_ASSERT((pC0[0] & 0x7f) == 0);
		//_ASSERT((pC1[0] & 0x7f) == 0);

		kernel_xor.setArg(0, pC_Out);
		kernel_xor.setArg(1, pC0);
		kernel_xor.setArg(2, N1);
		kernel_xor.setArg(3, N2);
		kernel_xor.setArg(4, N_);
		queue.enqueueNDRangeKernel(kernel_xor, cl::NullRange, cl::NDRange(N_), cl::NullRange);
#ifdef CL_TEST
		queue.enqueueReadBuffer(pC_Out, CL_TRUE, 0, sizeof(bool) * N_, test);
		for (int i = 0; i < 8; i++)
			cout << test[i] << " ";
		cout << endl;
#endif
		N_ <<= 1; //N=2*N
		N1 -= N_;
		N2 -= N_;
	}
};

cl::Buffer Decoder::getC(unsigned lambda){
	int pointer = 0;
	for (int i = 0; i < lambda; i++){
		pointer += (1 << (m - i));
	}
	return buffer_C0.at(lambda);
}

/*void Decoder::IterativelyCalcS_CPU(unsigned lambda, int phi)
{
	int Depth = 0;
	while (phi % (int)(pow(2, Depth)) == 0)
		if (Depth < lambda - 1)
			Depth++;
		else
			break;
	unsigned lambda0 = lambda - Depth;

	double *S1 = S[lambda0 - 1];
	unsigned N1 = 1 << (m - lambda0);
	double* S_;
	bool* C_l;
	if ((phi >> Depth) % 2 == 1)
	{
		S_ = S[lambda0];
		C_l = C[lambda0];

		for (int i = 0; i < N1; i++){
			if (C_l[i] == 0)
				S_[i] = S1[i] + S1[i + N1];
			else
				S_[i] = S1[i + N1] - S1[i];
		}
		S1 = S_;
		lambda0++;
		N1 = N1 / 2;
	
	};
	for (lambda0; lambda0 <= lambda; lambda0++)
	{
		S_ = S[lambda0];
		for (int i = 0; i < N1; i++){
			S_[i] = Q(S1[i+N1], S1[i]);
		}

		S1 = S_;
		N1 >>= 1;
	};
};


void Decoder::IterativelyUpdateC_CPU(unsigned lambda, unsigned phi)
{
	int Depth = 0;
	while (((phi + 1) % (2 ^ Depth))==0) Depth++;
	unsigned lambda0 = lambda - Depth;
	bool* pC_Out = C[lambda0];
	bool* pC0;
	unsigned N_ = (1 << (m - lambda));
	unsigned FinalLength = N_ << Depth;


	unsigned N1 = FinalLength - N_;
	unsigned N2 = FinalLength - 2 * N_;

	
	for (lambda; lambda > lambda0; lambda--)
	{
		pC0 = C[lambda];
		for (int i = 0; i < N_; i++){
			pC_Out[i + N2] = pC0[i] != pC_Out[i + N1];
		}
		N_ <<= 1; //N=2*N
		N1 -= N_;
		N2 -= N_;
	}
};*/