#include <vector>
#include <chrono>

#include <iostream>
#include <ctime>

#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/random.h>

#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

/* -- Security model --------------------------------------------------------- *
 *
 *  The old miner generated candidate seeds with curand (a PRNG seeded from a
 *  mere 64 bits) and then walked a +1 counter for 100k keys per thread.  That
 *  is catastrophically predictable: given one output seed, an attacker can
 *  derive ~100k neighbouring private keys.
 *
 *  This version replaces all of that with a ChaCha20 CSPRNG in counter mode:
 *
 *   * The host draws a fresh 32-byte master key and 8-byte nonce for every
 *     GPU on every run from the operating system CSPRNG (getrandom(2),
 *     falling back to /dev/urandom).
 *   * Every candidate seed is a fresh 32-byte slice of ChaCha20 keystream,
 *     i.e. ChaCha20(master_key, nonce, counter) with a unique counter per
 *     (run, GPU, kernel launch, thread, attempt).  No seed is ever repeated
 *     and none is predictable without the master key.
 *   * The master key is never printed, logged, or written anywhere; it is
 *     wiped from host memory after upload and lives only in GPU memory for
 *     the duration of the run.
 *
 *  ChaCha20 in counter mode keyed from OS entropy is the same construction
 *  the Linux kernel uses internally for /dev/urandom, so candidate seeds are
 *  computationally indistinguishable from uniform random bytes -- the found
 *  keypair is safe to use.
 * ------------------------------------------------------------------------- */

/* -- Types ----------------------------------------------------------------- */

// Master key material for the on-GPU CSPRNG.  NEVER print or log this.
typedef struct {
	unsigned char key[32];    // ChaCha20 key,   from the OS CSPRNG
	unsigned char nonce[8];   // ChaCha20 nonce, from the OS CSPRNG
} chacha20_key;

// The suffix constraint, uploaded to each GPU (host may widen/narrow it for
// smoke tests via VANITY_SMOKE=1).
typedef struct {
	unsigned char b28_mask;
	unsigned char b28_val;
	unsigned char b29_val;
	unsigned char b30_val;
	unsigned char b31_val;
} suffix_constraint;

typedef struct {
	int                 gpuCount;
	chacha20_key*       dev_ck[8];
	suffix_constraint*  dev_sc[8];
	unsigned long long  total_threads[8];
	int                 grid[8];
	int                 block[8];
	int*                dev_keys_found[8];
	int*                dev_executions[8];
	int*                dev_gpu_idx[8];
} config;

/* -- Prototypes, Because C++ ----------------------------------------------- */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_scan(const chacha20_key* ck, const suffix_constraint* sc,
                            unsigned long long base_seed_index,
                            int* keys_found, int* gpu, int* exec_count);
bool            run_self_test();
static void     get_os_entropy(unsigned char* buf, size_t len);
__host__ __device__ void chacha20_block(const chacha20_key* ck, unsigned long long counter, unsigned char out[64]);
__host__ __device__ bool pubkey_matches_suffix(const suffix_constraint* sc, const unsigned char* publick);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);

/* -- Entry Point ----------------------------------------------------------- */

int main(int argc, char const* argv[]) {
	ed25519_set_verbose(true);

	if (!run_self_test()) {
		fprintf(stderr, "FATAL: self test failed, refusing to mine\n");
		return 1;
	}

	config vanity;
	vanity_setup(vanity);
	vanity_run(vanity);
	return 0;
}

std::string getTimeStr(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
}

/* -- OS CSPRNG -------------------------------------------------------------- */

static void get_os_entropy(unsigned char* buf, size_t len) {
	size_t off = 0;
	while (off < len) {
		ssize_t r = getrandom(buf + off, len - off, 0);
		if (r > 0) {
			off += (size_t)r;
			continue;
		}
		if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;

		// Fallback for kernels without getrandom(2).
		int fd = open("/dev/urandom", O_RDONLY);
		if (fd < 0) {
			fprintf(stderr, "FATAL: no entropy source available (%s)\n", strerror(errno));
			exit(1);
		}
		while (off < len) {
			ssize_t n = read(fd, buf + off, len - off);
			if (n < 0 && errno == EINTR) continue;
			if (n <= 0) {
				fprintf(stderr, "FATAL: read(/dev/urandom) failed\n");
				exit(1);
			}
			off += (size_t)n;
		}
		close(fd);
		return;
	}
}

/* -- ChaCha20 (original djb variant: 64-bit counter, 8-byte nonce) ---------- *
 *
 *  Used as a CSPRNG in counter mode.  Reference vectors embedded in
 *  run_self_test() were generated independently and cross-checked against
 *  OpenSSL's chacha20 (RFC 8439 layout).
 * ------------------------------------------------------------------------- */

#define CHACHA_ROTL(v, c) ( ((v) << (c)) | ((v) >> (32 - (c))) )

#define CHACHA_QR(a, b, c, d)      \
	a += b; d ^= a; d = CHACHA_ROTL(d, 16); \
	c += d; b ^= c; b = CHACHA_ROTL(b, 12); \
	a += b; d ^= a; d = CHACHA_ROTL(d, 8);  \
	c += d; b ^= c; b = CHACHA_ROTL(b, 7)

__host__ __device__ void chacha20_block(const chacha20_key* ck, unsigned long long counter, unsigned char out[64]) {
	uint32_t st[16];

	st[0] = 0x61707865;
	st[1] = 0x3320646e;
	st[2] = 0x79622d32;
	st[3] = 0x6b206574;

	for (int i = 0; i < 8; ++i) {
		st[4 + i] = (uint32_t)ck->key[4 * i + 0]
		          | ((uint32_t)ck->key[4 * i + 1] <<  8)
		          | ((uint32_t)ck->key[4 * i + 2] << 16)
		          | ((uint32_t)ck->key[4 * i + 3] << 24);
	}

	st[12] = (uint32_t)(counter & 0xffffffffULL);
	st[13] = (uint32_t)(counter >> 32);

	st[14] = (uint32_t)ck->nonce[0]
	       | ((uint32_t)ck->nonce[1] <<  8)
	       | ((uint32_t)ck->nonce[2] << 16)
	       | ((uint32_t)ck->nonce[3] << 24);
	st[15] = (uint32_t)ck->nonce[4]
	       | ((uint32_t)ck->nonce[5] <<  8)
	       | ((uint32_t)ck->nonce[6] << 16)
	       | ((uint32_t)ck->nonce[7] << 24);

	uint32_t x[16];
	for (int i = 0; i < 16; ++i) x[i] = st[i];

	for (int r = 0; r < 10; ++r) {
		CHACHA_QR(x[0], x[4], x[ 8], x[12]);
		CHACHA_QR(x[1], x[5], x[ 9], x[13]);
		CHACHA_QR(x[2], x[6], x[10], x[14]);
		CHACHA_QR(x[3], x[7], x[11], x[15]);
		CHACHA_QR(x[0], x[5], x[10], x[15]);
		CHACHA_QR(x[1], x[6], x[11], x[12]);
		CHACHA_QR(x[2], x[7], x[ 8], x[13]);
		CHACHA_QR(x[3], x[4], x[ 9], x[14]);
	}

	for (int i = 0; i < 16; ++i) {
		uint32_t v = x[i] + st[i];
		out[4 * i + 0] = (unsigned char)(v);
		out[4 * i + 1] = (unsigned char)(v >>  8);
		out[4 * i + 2] = (unsigned char)(v >> 16);
		out[4 * i + 3] = (unsigned char)(v >> 24);
	}
}

/* -- Suffix matcher ---------------------------------------------------------- */

__host__ __device__ bool pubkey_matches_suffix(const suffix_constraint* sc, const unsigned char* publick) {
	return ((publick[28] & sc->b28_mask) == sc->b28_val)
	    && (publick[29] == sc->b29_val)
	    && (publick[30] == sc->b30_val)
	    && (publick[31] == sc->b31_val);
}

/* -- Self test ---------------------------------------------------------------- */

static bool hex2bin(unsigned char* out, const char* hex, size_t len) {
	for (size_t i = 0; i < len; ++i) {
		unsigned int b = 0;
		if (sscanf(hex + 2 * i, "%2x", &b) != 1) return false;
		out[i] = (unsigned char)b;
	}
	return true;
}

bool run_self_test() {
	bool ok = true;

	// 1) ChaCha20 against reference vectors (the reference implementation was
	//    cross-checked against OpenSSL's chacha20 / RFC 8439 test vectors).
	struct { const char* key; const char* nonce; unsigned long long ctr; const char* ks; } cv[] = {
		{ "3d1c3f2b0a1c2d3e4f5a6b7c8d9e0f1a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e",
		  "0102030405060708", 0ULL,
		  "3be147cc52bb35f13044daefb0191fb394496db9e46a7b3ff7c839cb14b751ed"
		  "e180a360bee5acc9f3c034de00e3980a221653c5010f255dde8e696cb750534a" },
		{ "3d1c3f2b0a1c2d3e4f5a6b7c8d9e0f1a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e",
		  "0102030405060708", 1ULL,
		  "10402bb51f45c74be2f937bb323bce3fd5ea8df08e1b07367617b89362830699"
		  "b1c27f90e8e66342b1cee054857e45e71e8e0623fab4ba1e1a010e581db2a1ab" },
		{ "3d1c3f2b0a1c2d3e4f5a6b7c8d9e0f1a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e",
		  "0102030405060708", 8589934592ULL, // 2^33: exercise the full 64-bit counter
		  "ff1a633a75b7e06f82c1187c291731bfd5cd221ac8ef385c887e1bf8a9167794"
		  "81ad24e80aea411f44103ebb6dde62df418793ff227948fc2347be82768fd9db" },
		{ "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
		  "000000090000004a", 1ULL,
		  "98f17a63810d031c3693986691316d20b7b1f9d11f8c57ae8376ad21a144f193"
		  "f2ee4aee752d1e62833da3edfd63feda04efb2a851229d21397d0cdb938a7eed" },
	};

	for (size_t i = 0; i < sizeof(cv) / sizeof(cv[0]); ++i) {
		chacha20_key ck;
		unsigned char expect[64], out[64];
		if (!hex2bin(ck.key, cv[i].key, 32) || !hex2bin(ck.nonce, cv[i].nonce, 8) ||
		    !hex2bin(expect, cv[i].ks, 64)) {
			fprintf(stderr, "self test: bad hex in chacha vector %zu\n", i);
			return false;
		}
		chacha20_block(&ck, cv[i].ctr, out);
		if (memcmp(out, expect, 64) != 0) {
			fprintf(stderr, "self test: chacha20 vector %zu MISMATCH\n", i);
			ok = false;
		}
	}

	// 2) ed25519 derivation, RFC 8032 section 7.1 test vectors 1 and 2.
	struct { const char* seed; const char* pub; } ev[] = {
		{ "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
		  "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a" },
		{ "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
		  "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c" },
	};

	for (size_t i = 0; i < sizeof(ev) / sizeof(ev[0]); ++i) {
		unsigned char seed[32], pub[32], priv[64], expect[32];
		if (!hex2bin(seed, ev[i].seed, 32) || !hex2bin(expect, ev[i].pub, 32)) {
			fprintf(stderr, "self test: bad hex in ed25519 vector %zu\n", i);
			return false;
		}
		ed25519_create_keypair(pub, priv, seed);
		if (memcmp(pub, expect, 32) != 0) {
			fprintf(stderr, "self test: ed25519 vector %zu MISMATCH\n", i);
			ok = false;
		}
	}

	// 3) The 30-bit constraint must be exactly the base64 bits of "+Ajax".
	{
		static const char alpha[] =
			"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
		const char* suffix = "+Ajax";
		unsigned long long bits = 0;
		for (int i = 0; i < 5; ++i) {
			const char* p = strchr(alpha, suffix[i]);
			if (p == NULL) {
				fprintf(stderr, "self test: bad suffix char '%c'\n", suffix[i]);
				return false;
			}
			bits = (bits << 6) | (unsigned long long)(p - alpha);
		}

		unsigned int want = ((unsigned int)(SUFFIX_B28_VAL & SUFFIX_B28_MASK) << 24)
		                  | ((unsigned int)SUFFIX_B29_VAL << 16)
		                  | ((unsigned int)SUFFIX_B30_VAL <<  8)
		                  | ((unsigned int)SUFFIX_B31_VAL);
		if ((unsigned int)(bits & 0x3FFFFFFFULL) != want) {
			fprintf(stderr, "self test: suffix constraint (%08x) != base64(\"%s\") bits (%08llx)\n",
				want, suffix, (unsigned long long)(bits & 0x3FFFFFFFULL));
			ok = false;
		}

		// The matcher must accept exactly that tail and reject neighbours.
		unsigned char pk[32] = {0};
		pk[28] = SUFFIX_B28_VAL | (unsigned char)~SUFFIX_B28_MASK;
		pk[29] = SUFFIX_B29_VAL;
		pk[30] = SUFFIX_B30_VAL;
		pk[31] = SUFFIX_B31_VAL;
		suffix_constraint sc;
		sc.b28_mask = SUFFIX_B28_MASK;
		sc.b28_val  = SUFFIX_B28_VAL;
		sc.b29_val  = SUFFIX_B29_VAL;
		sc.b30_val  = SUFFIX_B30_VAL;
		sc.b31_val  = SUFFIX_B31_VAL;
		if (!pubkey_matches_suffix(&sc, pk)) {
			fprintf(stderr, "self test: matcher rejected a matching tail\n");
			ok = false;
		}
		pk[31] ^= 1;
		if (pubkey_matches_suffix(&sc, pk)) {
			fprintf(stderr, "self test: matcher accepted a wrong tail\n");
			ok = false;
		}
	}

	if (ok) printf("Self test passed (chacha20 CSPRNG, ed25519 derivation, suffix constraint)\n");
	return ok;
}

/* -- Vanity Step Functions ------------------------------------------------- */

void vanity_setup(config &vanity) {
	printf("GPU: Initializing Memory\n");
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);
	if (gpuCount <= 0) {
		fprintf(stderr, "FATAL: no CUDA devices found\n");
		exit(1);
	}
	if (gpuCount > 8) gpuCount = 8;
	vanity.gpuCount = gpuCount;

	// Every GPU gets its own fresh CSPRNG master key, so counters never need
	// to be coordinated across devices.
	for (int i = 0; i < gpuCount; ++i) {
		cudaSetDevice(i);

		// Fetch Device Properties
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, i);

		// Calculate Occupancy
		int blockSize       = 0,
		    minGridSize     = 0,
		    maxActiveBlocks = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

		printf("GPU: %d (%s <%d, %d, %d>) -- W: %d, P: %d, TPB: %d, MTD: (%dx, %dy, %dz), MGS: (%dx, %dy, %dz)\n",
			i,
			device.name,
			blockSize,
			minGridSize,
			maxActiveBlocks,
			device.warpSize,
			device.multiProcessorCount,
		       	device.maxThreadsPerBlock,
			device.maxThreadsDim[0],
			device.maxThreadsDim[1],
			device.maxThreadsDim[2],
			device.maxGridSize[0],
			device.maxGridSize[1],
			device.maxGridSize[2]
		);

		// Launch with the *minimum grid that saturates the device*
		// (minGridSize blocks across all SMs).  maxActiveBlocks is
		// blocks-per-SM; launching with that as the grid size would leave
		// 55 of 56 SMs idle.
		vanity.grid[i]          = minGridSize;
		vanity.block[i]         = blockSize;
		vanity.total_threads[i] = (unsigned long long)minGridSize * blockSize;

		// CSPRNG master key: fresh OS entropy for every GPU on every run.
		// Deliberately NOT printed or logged anywhere.
		chacha20_key ck;
		get_os_entropy(ck.key,   sizeof(ck.key));
		get_os_entropy(ck.nonce, sizeof(ck.nonce));

		cudaMalloc((void **)&vanity.dev_ck[i], sizeof(chacha20_key));
		cudaMemcpy(vanity.dev_ck[i], &ck, sizeof(chacha20_key), cudaMemcpyHostToDevice);
		memset(&ck, 0, sizeof(ck)); // wipe the host copy after upload

		// Suffix constraint (VANITY_SMOKE=1 drops the masked byte for a quick
		// 24-bit pipeline smoke test).
		suffix_constraint sc;
		sc.b28_mask = SUFFIX_B28_MASK;
		sc.b28_val  = SUFFIX_B28_VAL;
		sc.b29_val  = SUFFIX_B29_VAL;
		sc.b30_val  = SUFFIX_B30_VAL;
		sc.b31_val  = SUFFIX_B31_VAL;
		const char* smoke = getenv("VANITY_SMOKE");
		if (smoke != NULL && smoke[0] == '1') {
			sc.b28_mask = 0x00;
			printf("SMOKE TEST: constraint widened to 24 bits (VANITY_SMOKE=1)\n");
		}

		cudaMalloc((void **)&vanity.dev_sc[i], sizeof(suffix_constraint));
		cudaMemcpy(vanity.dev_sc[i], &sc, sizeof(suffix_constraint), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&vanity.dev_keys_found[i], sizeof(int));
		cudaMalloc((void **)&vanity.dev_executions[i], sizeof(int));
		cudaMalloc((void **)&vanity.dev_gpu_idx[i],    sizeof(int));
		cudaMemcpy(vanity.dev_gpu_idx[i], &i, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(vanity.dev_keys_found[i], 0, sizeof(int));
		cudaMemset(vanity.dev_executions[i], 0, sizeof(int));

		printf("RNG: per-GPU ChaCha20 CSPRNG seeded from the OS CSPRNG (master key not logged)\n");
	}

	printf("END: Initializing Memory\n");
}

void vanity_run(config &vanity) {
	unsigned long long executions_total = 0;
	unsigned long long exec_last[8]     = {0};

	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		auto start = std::chrono::high_resolution_clock::now();

		unsigned long long executions_this_iteration = 0;

		// Every launch works on a disjoint counter range: launch `i` on a GPU
		// with T threads covers seeds
		//   [ i*T*ATTEMPTS_PER_EXECUTION, (i+1)*T*ATTEMPTS_PER_EXECUTION )
		// and thread `id` inside that launch covers
		//   [ base + id*ATTEMPTS_PER_EXECUTION, base + (id+1)*ATTEMPTS_PER_EXECUTION ).
		// Overlap is impossible.
		for (int g = 0; g < vanity.gpuCount; ++g) {
			cudaSetDevice(g);
			unsigned long long base =
				(unsigned long long)i * vanity.total_threads[g] * (unsigned long long)ATTEMPTS_PER_EXECUTION;
			vanity_scan<<<vanity.grid[g], vanity.block[g]>>>(
				vanity.dev_ck[g], vanity.dev_sc[g], base,
				vanity.dev_keys_found[g], vanity.dev_gpu_idx[g], vanity.dev_executions[g]);
		}

		// Synchronize while we wait for kernels to complete. I do not
		// actually know if this will sync against all GPUs, it might
		// just sync with the last `i`, but they should all complete
		// roughly at the same time and worst case it will just stack
		// up kernels in the queue to run.
		cudaDeviceSynchronize();
		auto finish = std::chrono::high_resolution_clock::now();

		int keys_found_total = 0;
		for (int g = 0; g < vanity.gpuCount; ++g) {
			cudaSetDevice(g);
			int keys_found  = 0;
			int exec_blocks = 0;
			cudaMemcpy(&keys_found, vanity.dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&exec_blocks, vanity.dev_executions[g], sizeof(int), cudaMemcpyDeviceToHost);
			keys_found_total += keys_found;

			unsigned long long this_exec =
				(unsigned long long)exec_blocks - exec_last[g];
			exec_last[g] = (unsigned long long)exec_blocks;
			executions_this_iteration += this_exec * (unsigned long long)ATTEMPTS_PER_EXECUTION;
		}
		executions_total += executions_this_iteration;

		// Print out performance Summary
		std::chrono::duration<double> elapsed = finish - start;
		printf("%s Iteration %d Attempts: %llu in %f at %fcps - Total Attempts %llu - keys found %d\n",
			getTimeStr().c_str(),
			i+1,
			executions_this_iteration,
			elapsed.count(),
			executions_this_iteration / elapsed.count(),
			executions_total,
			keys_found_total
		);

		if ( keys_found_total >= STOP_AFTER_KEYS_FOUND ) {
			printf("Enough keys found, Done! \n");
			exit(0);
		}
	}

	printf("Iterations complete, Done!\n");
}

/* -- CUDA Vanity Functions ------------------------------------------------- */

void __global__ vanity_scan(const chacha20_key* ck, const suffix_constraint* sc,
                            unsigned long long base_seed_index,
                            int* keys_found, int* gpu, int* exec_count) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);

        atomicAdd(exec_count, 1);

	// Local Kernel State
	ge_p3 A;
	unsigned char publick[32]  = {0};
	unsigned char privatek[64] = {0};
	unsigned char ks[64]       = {0};   // current ChaCha20 keystream block (2 seeds)
	sha512_context md;

	// This thread's candidate seeds are keystream slices starting at
	// seed_index = base + id*ATTEMPTS_PER_EXECUTION, one per attempt.
	unsigned long long thread_base =
		base_seed_index + (unsigned long long)id * (unsigned long long)ATTEMPTS_PER_EXECUTION;

	for (int attempts = 0; attempts < ATTEMPTS_PER_EXECUTION; ++attempts) {
		// -- CSPRNG candidate seed ------------------------------------------
		// ChaCha20 counter mode: two 32-byte seeds per 64-byte keystream
		// block.  (iteration, thread, attempt) triples map to unique block
		// counters, so every seed is fresh, uniform and unpredictable
		// without the master key.
		unsigned long long s = thread_base + (unsigned long long)attempts;
		if ((attempts & 1) == 0) {
			chacha20_block(ck, s >> 1, ks);
		}
		const unsigned char* seed = ks + (attempts & 1) * 32;

		// sha512_init Inlined
		md.curlen   = 0;
		md.length   = 0;
		md.state[0] = UINT64_C(0x6a09e667f3bcc908);
		md.state[1] = UINT64_C(0xbb67ae8584caa73b);
		md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
		md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
		md.state[4] = UINT64_C(0x510e527fade682d1);
		md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
		md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
		md.state[7] = UINT64_C(0x5be0cd19137e2179);

		// sha512_update inlined (always a 32-byte input)
		const unsigned char *in = seed;
		for (size_t i = 0; i < 32; i++) {
			md.buf[i + md.curlen] = in[i];
		}
		md.curlen += 32;

		// sha512_final inlined (single compress for a 32-byte input)
		md.length += md.curlen * UINT64_C(8);
		md.buf[md.curlen++] = (unsigned char)0x80;

		while (md.curlen < 120) {
			md.buf[md.curlen++] = (unsigned char)0;
		}

		STORE64H(md.length, md.buf+120);

		// Inline sha512_compress
		uint64_t S[8], W[80], t0, t1;
		int i;

		/* Copy state into S */
		for (i = 0; i < 8; i++) {
			S[i] = md.state[i];
		}

		/* Copy the state into 1024-bits into W[0..15] */
		for (i = 0; i < 16; i++) {
			LOAD64H(W[i], md.buf + (8*i));
		}

		/* Fill W[16..79] */
		for (i = 16; i < 80; i++) {
			W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];
		}

		/* Compress */
		#define RND(a,b,c,d,e,f,g,h,i) \
		t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
		t1 = Sigma0(a) + Maj(a, b, c);\
		d += t0; \
		h  = t0 + t1;

		for (i = 0; i < 80; i += 8) {
			RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);
			RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);
			RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);
			RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);
			RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);
			RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);
			RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);
			RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);
		}

		#undef RND

		/* Feedback */
		for (i = 0; i < 8; i++) {
			md.state[i] = md.state[i] + S[i];
		}

		// We can now output our finalized bytes into the output buffer.
		for (i = 0; i < 8; i++) {
			STORE64H(md.state[i], privatek+(8*i));
		}

		// ed25519 Hash Clamping
		privatek[0]  &= 248;
		privatek[31] &= 63;
		privatek[31] |= 64;

		// ed25519 curve multiplication to extract a public key.
		ge_scalarmult_base(&A, privatek);
		ge_p3_tobytes(publick, &A);

		if (pubkey_matches_suffix(sc, publick)) {
			atomicAdd(keys_found, 1);

			printf("GPU %d MATCH ,", *gpu);
			for(int n=0; n<32; n++) {
				printf("%02x",(unsigned char)seed[n]);
			}
			printf("\n");
			printf("[");
			for(int n=0; n<32; n++) {
				printf("%02x",(unsigned char)publick[n]);
			}
			printf("]\n");
		}
	}
}

bool __device__ b58enc(
	char    *b58,
       	size_t  *b58sz,
       	uint8_t *data,
       	size_t  binsz
) {
	// Base58 Lookup Table
	const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

	const uint8_t *bin = data;
	int carry;
	size_t i, j, high, zcount = 0;
	size_t size;

	while (zcount < binsz && !bin[zcount])
		++zcount;

	size = (binsz - zcount) * 138 / 100 + 1;
	uint8_t buf[256];
	memset(buf, 0, size);

	for (i = zcount, high = size - 1; i < binsz; ++i, high = j)
	{
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
		{
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (!j) {
				// Otherwise j wraps to maxint which is > high
				break;
			}
		}
	}

	for (j = 0; j < size && !buf[j]; ++j);

	if (*b58sz <= zcount + size - j) {
		*b58sz = zcount + size - j + 1;
		return false;
	}

	if (zcount) memset(b58, '1', zcount);
	for (i = zcount; j < size; ++i, ++j) b58[i] = b58digits_ordered[buf[j]];

	b58[i] = '\0';
	*b58sz = i + 1;

	return true;
}
