"""Compiled/eager checkpoints must load strictly without changing weights."""
import pytest
import torch
from flash_rl.agents.utils.network import Network


@pytest.mark.parametrize('source_compiled', [False, True])
@pytest.mark.parametrize('target_compiled', [False, True])
def test_checkpoint_compile_modes(tmp_path, source_compiled, target_compiled):
    torch.manual_seed(1)
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    wrapped = torch.compile(source, backend='eager') if source_compiled else source
    optimizer = torch.optim.Adam(source.parameters())
    source(torch.randn(8, 3)).square().mean().backward()
    optimizer.step()
    path = str(tmp_path / 'net.pt')
    Network(wrapped, optimizer=optimizer, update_step=7).save(path)
    before = (tmp_path / 'net.pt').read_bytes()
    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    target_optimizer = torch.optim.Adam(target.parameters())
    bundle = Network(torch.compile(target, backend='eager') if target_compiled else target,
                     optimizer=target_optimizer)
    bundle.load(path)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    assert bundle.update_step == 7
    assert len(target_optimizer.state) == len(optimizer.state)
    assert (tmp_path / 'net.pt').read_bytes() == before
    source.eval(); target.eval()
    x = torch.randn(8, 3)
    torch.testing.assert_close(target(x), source(x), rtol=0, atol=0)


def test_incompatible_shapes_still_fail(tmp_path):
    path = str(tmp_path / 'net.pt')
    Network(torch.compile(torch.nn.Linear(3, 4), backend='eager')).save(path)
    with pytest.raises(RuntimeError, match='size mismatch'):
        Network(torch.nn.Linear(3, 5)).load(path, load_optimizer=False)
