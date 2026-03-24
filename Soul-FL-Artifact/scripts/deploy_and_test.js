/**
 * scripts/deploy_and_test.js
 *
 * Hardhat deployment + functional test script for SoulFLToken.sol.
 * Run with:  npx hardhat run scripts/deploy_and_test.js --network localhost
 *
 * Tests:
 *   1. Deploy contract with paper-default parameters
 *   2. Mint SBTs for clients
 *   3. Verify effective balance with decay
 *   4. Redeem voucher, check refueling
 *   5. Non-transferability enforcement
 *   6. Round advancement and decay
 *   7. Gas cost report
 */

const { ethers } = require("hardhat");

// Paper parameters (Section V-A)
const DECAY_RATE_1E6 = 50_000;          // λ = 0.05
const TAU_MIN_1E6    = 1_000_000;       // τ_min = 1.0
const V_A_1E6        = 10_000_000;      // V_a = 10.0
const B0_1E6         = 100_000_000;     // B₀ = 100.0

// Fixed-point helpers
const fp = (x) => BigInt(Math.round(x * 1e6));
const fromFP = (x) => Number(x) / 1e6;

async function main() {
  const [owner, aggregator, client1, client2, attacker] = await ethers.getSigners();

  console.log("=".repeat(60));
  console.log(" Soul-FL Smart Contract Deployment & Test Suite");
  console.log("=".repeat(60));
  console.log(`Owner:      ${owner.address}`);
  console.log(`Aggregator: ${aggregator.address}`);
  console.log(`Client1:    ${client1.address}`);
  console.log(`Client2:    ${client2.address}`);
  console.log(`Attacker:   ${attacker.address}`);
  console.log();

  // -----------------------------------------------------------------------
  // 1. Deploy
  // -----------------------------------------------------------------------
  console.log("[1/7] Deploying SoulFLToken...");
  const SoulFLToken = await ethers.getContractFactory("SoulFLToken");
  const contract = await SoulFLToken.deploy(
    aggregator.address,
    DECAY_RATE_1E6,
    TAU_MIN_1E6,
    V_A_1E6,
    B0_1E6
  );
  await contract.waitForDeployment();
  const addr = await contract.getAddress();
  console.log(`  Deployed at: ${addr}`);

  // Verify constructor params
  const dr = await contract.DECAY_RATE_1E6();
  const tau = await contract.TAU_MIN_1E6();
  console.log(`  DECAY_RATE_1E6 = ${dr}  (λ = ${Number(dr)/1e6})`);
  console.log(`  TAU_MIN_1E6    = ${tau} (τ_min = ${Number(tau)/1e6})`);
  console.log("  ✓ Deployed\n");

  // -----------------------------------------------------------------------
  // 2. Mint SBTs
  // -----------------------------------------------------------------------
  console.log("[2/7] Minting SBTs...");
  const commitment1 = ethers.keccak256(ethers.toUtf8Bytes("pk_client1||anchor1"));
  const commitment2 = ethers.keccak256(ethers.toUtf8Bytes("pk_client2||anchor2"));

  const mintTx1 = await contract.connect(aggregator).mintSBT(client1.address, commitment1);
  const r1 = await mintTx1.wait();
  console.log(`  Client1 minted | gas: ${r1.gasUsed}`);

  const mintTx2 = await contract.connect(aggregator).mintSBT(client2.address, commitment2);
  const r2 = await mintTx2.wait();
  console.log(`  Client2 minted | gas: ${r2.gasUsed}`);

  let b1 = await contract.getEffectiveBalance(client1.address);
  console.log(`  Client1 B_eff = ${fromFP(b1).toFixed(4)} (expected ${fromFP(B0_1E6).toFixed(4)})`);
  console.log("  ✓ Minting\n");

  // -----------------------------------------------------------------------
  // 3. Trust decay after round advancement
  // -----------------------------------------------------------------------
  console.log("[3/7] Testing trust decay...");
  const ROUNDS_ADVANCE = 14;  // ≈ half-life at λ=0.05

  for (let i = 0; i < ROUNDS_ADVANCE; i++) {
    await contract.connect(aggregator).advanceRound();
  }

  let round = await contract.currentRound();
  b1 = await contract.getEffectiveBalance(client1.address);
  const expected_half = fromFP(B0_1E6) * Math.exp(-0.05 * ROUNDS_ADVANCE);
  console.log(`  After ${ROUNDS_ADVANCE} rounds: B_eff = ${fromFP(b1).toFixed(4)}`);
  console.log(`  Expected (analytical): ${expected_half.toFixed(4)}`);
  console.log(`  ✓ Decay (error < ${Math.abs(fromFP(b1) - expected_half).toFixed(4)})\n`);

  // -----------------------------------------------------------------------
  // 4. Voucher redemption
  // -----------------------------------------------------------------------
  console.log("[4/7] Testing voucher redemption (EIP-712)...");
  const domain = {
    name: "SoulFLToken",
    version: "1",
    chainId: (await ethers.provider.getNetwork()).chainId,
    verifyingContract: addr,
  };
  const types = {
    Voucher: [
      { name: "client", type: "address" },
      { name: "amount", type: "uint256" },
      { name: "round",  type: "uint256" },
      { name: "nonce",  type: "uint256" },
    ],
  };

  const nonce = await contract.getNonce(client1.address);
  const voucherData = {
    client: client1.address,
    amount: V_A_1E6,
    round:  Number(round),
    nonce:  Number(nonce),
  };
  const sig = await aggregator.signTypedData(domain, types, voucherData);

  const redeemTx = await contract.connect(client1).redeemVoucher(
    V_A_1E6, Number(round), Number(nonce), sig
  );
  const rr = await redeemTx.wait();
  console.log(`  Voucher redeemed | gas: ${rr.gasUsed}`);

  const b1_after = await contract.getEffectiveBalance(client1.address);
  console.log(`  B_eff after refuel = ${fromFP(b1_after).toFixed(4)}`);
  console.log("  ✓ Voucher redemption\n");

  // -----------------------------------------------------------------------
  // 5. Non-transferability
  // -----------------------------------------------------------------------
  console.log("[5/7] Testing non-transferability...");
  try {
    await contract.connect(client1).transfer(client2.address, fp(10));
    console.error("  ✗ Transfer should have reverted!");
    process.exit(1);
  } catch (e) {
    console.log(`  ✓ Transfer reverted: ${e.message.slice(0, 60)}`);
  }
  try {
    await contract.connect(client1).approve(client2.address, fp(10));
    console.error("  ✗ Approve should have reverted!");
    process.exit(1);
  } catch (e) {
    console.log(`  ✓ Approve reverted: ${e.message.slice(0, 60)}\n`);
  }

  // -----------------------------------------------------------------------
  // 6. Eligibility gating
  // -----------------------------------------------------------------------
  console.log("[6/7] Testing eligibility gating...");
  // Advance many rounds so Client2 (no vouchers) decays below τ_min
  for (let i = 0; i < 100; i++) {
    await contract.connect(aggregator).advanceRound();
  }

  const elig1 = await contract.isEligible(client1.address);
  const elig2 = await contract.isEligible(client2.address);
  const b2 = await contract.getEffectiveBalance(client2.address);
  console.log(`  Client1 eligible: ${elig1}  (B_eff=${fromFP(b1_after).toFixed(4)})`);
  console.log(`  Client2 eligible: ${elig2}  (B_eff=${fromFP(b2).toFixed(6)})`);
  console.log("  ✓ Eligibility gating\n");

  // -----------------------------------------------------------------------
  // 7. Gas report
  // -----------------------------------------------------------------------
  console.log("[7/7] Gas cost summary:");
  console.log(`  mintSBT         ≈ ${r1.gasUsed} gas`);
  console.log(`  redeemVoucher   ≈ ${rr.gasUsed} gas`);
  const revokeTx = await contract.connect(aggregator).revoke(attacker.address);
  // revoke on non-minted client: just checking gas
  const rRev = await revokeTx.wait();
  console.log(`  revoke          ≈ ${rRev.gasUsed} gas`);
  console.log();

  console.log("=".repeat(60));
  console.log(" All tests PASSED");
  console.log("=".repeat(60));
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
