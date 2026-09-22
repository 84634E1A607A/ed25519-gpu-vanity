#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int const MAX_ITERATIONS = 100000;

// Stop as soon as a single matching key has been found.
static int const STOP_AFTER_KEYS_FOUND = 1;

// how many times a gpu thread generates a public key in one go
// NOTE: must be an even number -- two candidate seeds come out of each
// ChaCha20 keystream block.
static int const ATTEMPTS_PER_EXECUTION = 10000;

// -----------------------------------------------------------------------------
// Target: the base64 encoding of the ssh-ed25519 wire blob
//
//   blob = string("ssh-ed25519") || string(32-byte public key)
//
// is exactly 51 bytes, so its base64 form is 68 chars with NO padding.  The
// last 5 characters cover exactly the final 30 bits of the blob, i.e. the
// final 30 bits of the public key:
//
//   "+Ajax" = '+'=62  'A'=0  'j'=35  'a'=26  'x'=49
//           = 111110 000000 100011 011010 110001
//
// which pins the tail of the public key to:
//
//   pubkey[28] & 0x3F == 0x3E
//   pubkey[29] == 0x02
//   pubkey[30] == 0x36
//   pubkey[31] == 0xB1
//
// i.e. big-endian (pubkey[28..31] & 0x3FFFFFFF) == 0x3E0236B1.
// That is a 30-bit constraint: one match per ~2^30 candidate keys.
// -----------------------------------------------------------------------------

static const unsigned char SUFFIX_B28_MASK = 0x3F;   // low 6 bits of pubkey[28]
static const unsigned char SUFFIX_B28_VAL  = 0x3E;
static const unsigned char SUFFIX_B29_VAL  = 0x02;
static const unsigned char SUFFIX_B30_VAL  = 0x36;
static const unsigned char SUFFIX_B31_VAL  = 0xB1;

#endif
